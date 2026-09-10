#!/usr/bin/env python3
"""A masked-autoencoder scene embedding trained on OUR route tensors, for the DICE-style baseline of
run_av_baselines.py (Farid, Schleede, Huang, Heckman, "Foundation models for rapid autonomy validation", ICRA 2025).

The paper pre-trains a 34M-parameter MAE on 14 million proprietary 10 s snippets (tracks, traffic signals, road
polylines) and mean-pools the ego-state embedding of a scenario as its 64-d representation; none of that is
released. This is the same recipe at the scale of the Bench2Drive bank: the 2,656 RelGraph windows of the 220
routes (encoder/harness/b2d_relgraph_v2.npz — ego 12 x 6 + command, 48 agent tracks 12 x 8, 128 lane polylines
10 x 4 + 2), tokens = ego time steps, agent tracks and lane polylines projected to D = 64, a random mask of ratio
r = 0.5 over the valid tokens replaced by a zero vector (Sec. III-C), a 4-layer transformer encoder, per-type
linear decoders, L1 reconstruction on every valid token with equal type weights (Sec. III-E with lambda = 1 and
r_loss = 1), 30 epochs of AdamW 3e-4. The window embedding is the mean of the ego-step tokens (Sec. VI-A), the
route embedding the mean over its windows. No response, planner or scenario-type label is read: this is
unsupervised on scene content only. Writes data/features/eval_dice_mae.npz (the 64-d ego embedding used for
clustering) and eval_dice_mae_pooled.npz (the pooled ego / track / road embeddings, 192-d, that the paper's
difficulty head takes as input; Sec. IV-B) in the load_features format.

    CUDA_VISIBLE_DEVICES=1 python experiments/dice_mae.py
"""
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
NPZ = ROOT / 'encoder' / 'harness' / 'b2d_relgraph_v2.npz'
OUT = ROOT / 'data' / 'features' / 'eval_dice_mae.npz'
D, LAYERS, HEADS, EPOCHS, BS, LR, MASK, SEED = 64, 4, 4, 30, 64, 3e-4, 0.5, 0


def load():
    z = np.load(NPZ, allow_pickle=True)
    ids = [str(x) for x in z['item_id']]
    rid = np.array([re.match(r'route_(\d+)_(\d+)$', s).group(1) for s in ids])
    ego = np.concatenate([z['ego'].astype(np.float32), np.repeat(z['command'].astype(np.float32)[:, None, :], 12, 1)], -1)  # (W,12,10)
    ag = z['agents'].astype(np.float32)                                     # (W,48,12,8)
    am = z['agent_mask']                                                    # (W,48,12)
    ag = ag * am[..., None]
    agv = am.any(-1)                                                        # (W,48) a track with any valid step
    ag = ag.reshape(len(ag), 48, -1)                                        # (W,48,96)
    ln = np.concatenate([z['lanes'].astype(np.float32).reshape(len(ag), 128, -1), z['lane_feat'].astype(np.float32)], -1)  # (W,128,42)
    lm = z['lane_mask']                                                     # (W,128)
    ln = ln * lm[..., None]

    def stdz(x, valid):
        mu = x[valid].mean(0)
        sd = x[valid].std(0) + 1e-6
        return ((x - mu) / sd) * valid[..., None]
    ego = stdz(ego, np.ones(ego.shape[:2], bool))
    ag = stdz(ag, agv)
    ln = stdz(ln, lm)
    return rid, ego, ag, agv, ln, lm


class MAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.p_ego, self.p_ag, self.p_ln = nn.Linear(10, D), nn.Linear(96, D), nn.Linear(42, D)
        self.type_emb = nn.Parameter(torch.zeros(3, D))
        self.pos_ego = nn.Parameter(torch.zeros(12, D))
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(D, HEADS, 2 * D, dropout=0.0, batch_first=True), LAYERS)
        self.d_ego, self.d_ag, self.d_ln = nn.Linear(D, 10), nn.Linear(D, 96), nn.Linear(D, 42)
        nn.init.normal_(self.type_emb, std=0.02)
        nn.init.normal_(self.pos_ego, std=0.02)

    def tokens(self, ego, ag, ln):
        t_ego = self.p_ego(ego) + self.type_emb[0] + self.pos_ego
        t_ag = self.p_ag(ag) + self.type_emb[1]
        t_ln = self.p_ln(ln) + self.type_emb[2]
        return torch.cat([t_ego, t_ag, t_ln], 1)                            # (B, 12+48+128, D)

    def forward(self, ego, ag, agv, ln, lm, mask=None):
        tok = self.tokens(ego, ag, ln)
        valid = torch.cat([torch.ones(ego.shape[:2], dtype=torch.bool, device=ego.device), agv, lm], 1)
        if mask is not None:
            tok = tok * (~mask)[..., None]                                  # masked token -> zero vector (Sec. III-C)
        h = self.enc(tok, src_key_padding_mask=~valid)
        return h, valid

    def decode(self, h):
        return self.d_ego(h[:, :12]), self.d_ag(h[:, 12:60]), self.d_ln(h[:, 60:])


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    rid, ego, ag, agv, ln, lm = load()
    W = len(rid)
    T = lambda x: torch.tensor(x, device=dev)
    ego_t, ag_t, agv_t, ln_t, lm_t = T(ego), T(ag), T(agv), T(ln), T(lm)
    m = MAE().to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=0.01)
    rng = np.random.default_rng(SEED)
    print(f'[dice_mae] {W} windows, {len(set(rid))} routes, {sum(p.numel() for p in m.parameters())} parameters, device {dev}', flush=True)
    for ep in range(EPOCHS):
        m.train()
        order = rng.permutation(W)
        tot = nb = 0
        for i0 in range(0, W, BS):
            idx = T(order[i0:i0 + BS])
            e, a, av, l, lv = ego_t[idx], ag_t[idx], agv_t[idx], ln_t[idx], lm_t[idx]
            valid = torch.cat([torch.ones(e.shape[:2], dtype=torch.bool, device=dev), av, lv], 1)
            mask = (torch.rand(valid.shape, device=dev) < MASK) & valid
            h, _ = m(e, a, av, l, lv, mask)
            r_e, r_a, r_l = m.decode(h)
            loss = ((r_e - e).abs().mean(-1)).mean() \
                + ((r_a - a).abs().mean(-1) * av).sum() / av.sum().clamp(min=1) \
                + ((r_l - l).abs().mean(-1) * lv).sum() / lv.sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            tot += float(loss)
            nb += 1
        if ep % 5 == 0 or ep == EPOCHS - 1:
            print(f'[dice_mae] epoch {ep:2d} loss {tot / nb:.4f}', flush=True)
    m.eval()
    with torch.no_grad():                                                   # no masking at inference (Sec. III-C)
        emb, pooled = [], []
        for i in range(0, W, 256):
            h, valid = m(ego_t[i:i + 256], ag_t[i:i + 256], agv_t[i:i + 256], ln_t[i:i + 256], lm_t[i:i + 256])
            emb.append(h[:, :12].mean(1))                                   # the ego-state embedding (Sec. VI-A)
            av, lv = agv_t[i:i + 256].float(), lm_t[i:i + 256].float()      # the pooled ego / track / road embeddings
            pooled.append(torch.cat([h[:, :12].mean(1),                     # that feed the difficulty head (Sec. IV-B)
                                     (h[:, 12:60] * av[..., None]).sum(1) / av.sum(1, keepdim=True).clamp(min=1),
                                     (h[:, 60:] * lv[..., None]).sum(1) / lv.sum(1, keepdim=True).clamp(min=1)], 1))
        emb = torch.cat(emb).cpu().numpy()
        pooled = torch.cat(pooled).cpu().numpy()
    routes = sorted(set(rid))
    names = np.array([f'route_{r}' for r in routes])
    stats = np.stack([emb[rid == r].mean(0) for r in routes]).astype(np.float32)
    np.savez(OUT, names=names, stats=stats)
    head = np.stack([pooled[rid == r].mean(0) for r in routes]).astype(np.float32)
    np.savez(OUT.with_name('eval_dice_mae_pooled.npz'), names=names, stats=head)
    print(f'[dice_mae] wrote {OUT}: {stats.shape}, per-dim SD mean {stats.std(0).mean():.3f}; pooled head input {head.shape}', flush=True)


if __name__ == '__main__':
    t0 = time.time()
    main()
    print(f'[dice_mae] done in {time.time() - t0:.0f} s')

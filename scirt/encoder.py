"""The difficulty-supervised interaction encoder (the paper's "Ours" row).

A scene is a padded sequence of 6 s windows; each window holds up to 48 agent
box-tracks and the ego track in the window-anchor frame. Per window, agent
tracks and the ego track are GRU-encoded into tokens and mixed by a small
transformer; the ego token is read out, a learned attention pool aggregates
windows to one route embedding, ego kinematic statistics are embedded and
concatenated (the "kin-embedded" fusion), and a linear head emits scalar
difficulty b_tilde.

Training signal: P(planner j fails scene i) = sigmoid(b_tilde_i - theta_j),
BCE over the full response panel, theta fitted per training fold by the a==1
calibration below and then frozen. The encoder never sees the gold anchor.

Torch is imported lazily so the CPU-pinned evaluation package does not pay for
it; training runs on GPU and is *not* bit-reproducible across devices — see
train/train_encoder_b2d.py for what is and is not pinned.
"""

import numpy as np


def rasch(Y, it=400, seed=0):
    """Fold-internal a==1 calibration used only to freeze theta for training.

    Same MAP objective family as scirt.irt but with the discrimination fixed:
    split-half reliability of fitted log-a on this panel is 0.03-0.15, so a is
    noise here and the training target keeps the Rasch geometry. Y is fail=1
    with NaN for missing cells. Returns (theta, b), b centred.
    """
    import torch

    torch.manual_seed(seed)
    Ym = ~np.isnan(Y)
    Yt = torch.tensor(np.nan_to_num(Y), dtype=torch.float32)
    W = torch.tensor(Ym.astype(np.float32))
    J, N = Y.shape
    th = torch.zeros(J, requires_grad=True)
    bb = torch.zeros(N, requires_grad=True)
    opt = torch.optim.Adam([th, bb], lr=0.05)
    for _ in range(it):
        p = torch.sigmoid(bb[None, :] - th[:, None])
        nll = (-(Yt * torch.log(p + 1e-7)
                 + (1 - Yt) * torch.log(1 - p + 1e-7)) * W).sum() / W.sum()
        (nll + 1e-2 * th.pow(2).mean() + 1e-3 * bb.pow(2).mean()).backward()
        opt.step()
        opt.zero_grad()
    c = float(bb.detach().mean())
    return th.detach().numpy() + c, bb.detach().numpy() - c


def build(torch, nn, d=64, heads=4, depth=2, kin_dim=25, dropout=0.15):
    """The encoder. kin_dim is the width of the route-level kinematic input."""

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.ag_gru = nn.GRU(8, d // 2, batch_first=True, bidirectional=True)
            self.eg_gru = nn.GRU(6, d // 2, batch_first=True, bidirectional=True)
            self.cmd_in = nn.Linear(4, d)
            layer = nn.TransformerEncoderLayer(
                d, heads, d * 2, dropout=dropout, batch_first=True, norm_first=True)
            self.trunk = nn.TransformerEncoder(layer, depth)
            self.wq = nn.Linear(d, 1)                     # window attention pool
            self.kin_in = nn.Sequential(
                nn.Linear(kin_dim, d), nn.ReLU(), nn.Linear(d, d))
            self.head = nn.Sequential(nn.LayerNorm(2 * d), nn.Linear(2 * d, 1))

        def forward(self, ag, am, eg, cm, wmask, kf=None):
            R, W = wmask.shape
            B = R * W
            h, _ = self.ag_gru(ag.reshape(B, 48, 12, 8).reshape(B * 48, 12, 8))
            atok = h.mean(1).reshape(B, 48, -1)
            he, _ = self.eg_gru(eg.reshape(B, 12, 6))
            q = he.mean(1) + self.cmd_in(cm.reshape(B, 4))
            toks = torch.cat([q[:, None], atok], 1)
            pad = torch.cat(
                [torch.zeros(B, 1, dtype=torch.bool, device=ag.device),
                 ~am.reshape(B, 48, 12).any(-1)], 1)
            z = self.trunk(toks, src_key_padding_mask=pad)[:, 0].reshape(R, W, -1)
            att = self.wq(z).squeeze(-1).masked_fill(~wmask, -1e9).softmax(-1)
            zr = (z * att[..., None]).sum(1)
            zr = torch.cat([zr, self.kin_in(kf)], -1)
            return self.head(zr).squeeze(-1), att

    return Net()

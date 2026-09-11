"""Window-level visual-temporal branch of the route encoder (the `--visual-window` arm, 2026-09-12).

Design (the user's specification):
    z_t = E_vision(I_t)                        frozen DINOv3 ViT-L/16 CLS token per frame, the three cameras
                                               concatenated -> 3072-d; T = 12 frames per window at 2 Hz, the
                                               SAME 10-Hz frames 20w + 5t the track window w is built from
    h_1..h_T = E_temporal(P z_1 + e_1, ..., P z_T + e_T)
                                               P: Linear(3072 -> d_v) [optionally after a per-draw PCA fitted on
                                               the training windows], e_t: learned temporal position (T x d_v),
                                               E_temporal: 2-layer bidirectional TransformerEncoder, d_v = 128,
                                               4 heads, FFN 2 d_v, dropout 0.1 (the rollout is recorded, so
                                               every frame may attend to every other frame)
    v_w = mean_t h_t                           over the valid frames of the window
    a_w = E_motion(window w)                   the R2-noLane window encoder's 64-d z_w of the same window
    r_w = MLP([v_w ; a_w])                     Linear(d_v + 64 -> d_v) SiLU Linear(d_v -> d_v)
    r_i = [mean_w r_w ; max_w r_w]             route pooling over the windows (2 d_v)
    mu_i = Head(r_i)                           LayerNorm(2 d_v) Linear(2 d_v -> 64) SiLU Linear(64 -> 1)
The route-level ego-sequence branch of the encoder of record is NOT used in this arm (the specification fuses
at the window level only). Everything else (loss, sigma_r, draws, seeds, standardisation on training data
only) is the harness's.
"""
import re

import numpy as np

T_WIN = 12          # frames per window (2 Hz)
VSTRIDE = 5         # the stored visual frames are every 5th 10-Hz frame


def load_window_visual(visual_dir, ids, tokens='cls', views='all'):
    """Per window row of the graph tensors, the T_WIN per-frame visual features: (N, T_WIN, 3 * C) float32 and
    a validity mask (N, T_WIN). Row `route_<r>_<w>` uses stored visual rows 4w + t (= 10-Hz frames 20w + 5t)."""
    cache = {}
    N = len(ids)
    first = None
    X = M = None
    for n, s in enumerate(ids):
        r, w = re.match(r'route_(\d+)_(\d+)$', s).groups()
        if r not in cache:
            z = np.load(f'{visual_dir}/route_{r}.npz')
            vsel = slice(0, 1) if views == 'front' else slice(None)                  # 'front': rgb_front only
            parts = [z['cls'][:, vsel].astype(np.float32)] + ([z['patch'][:, vsel].astype(np.float32)] if tokens == 'cls+patch' else [])
            feat = np.concatenate(parts, -1).reshape(len(z['frame']), -1)          # (Lv, n_views x C)
            assert (z['frame'] == np.arange(len(z['frame'])) * VSTRIDE).all(), 'visual frames must be every 5th frame'
            cache[r] = feat
        feat = cache[r]
        if X is None:
            X = np.zeros((N, T_WIN, feat.shape[1]), np.float32)
            M = np.zeros((N, T_WIN), bool)
        lo = 4 * int(w)
        k = min(T_WIN, len(feat) - lo)
        assert k >= 1, f'{s}: no visual frame for this window (needs stored row {lo}, the route has {len(feat)} rows)'
        X[n, :k] = feat[lo:lo + k]
        M[n, :k] = True
    n_partial = int((M.sum(1) < T_WIN).sum())
    if n_partial:
        print(f'[visual_window] WARNING: {n_partial} of {N} windows have fewer than {T_WIN} visual frames (masked)', flush=True)
    return X, M


def check_feature_cache(feature_dir, keys):
    """Every route file of a frozen-feature cache must carry the SAME value for each of `keys` (model / stride / views,
    or ckpt / stride_frames for the motion cache); returns that provenance dict, raises on a mixed cache."""
    import glob
    files = sorted(glob.glob(f'{feature_dir}/route_*.npz'))
    assert files, f'{feature_dir}: no route_*.npz'
    prov = None
    for f in files:
        z = np.load(f)
        cur = {k: (tuple(str(v) for v in z[k]) if np.ndim(z[k]) else str(z[k])) for k in keys if k in z.files}
        if prov is None:
            prov = cur
        assert cur == prov, f'{f}: feature cache is mixed: {cur} vs {prov}'
    return {'dir': feature_dir, 'n_files': len(files), **prov}


def build_visual_window(torch, nn, in_dim, d_v=128, d_track=64, layers=2, heads=4, dropout=0.1):
    class VisualWindow(nn.Module):
        def __init__(self):
            super().__init__()
            self.P = nn.Linear(in_dim, d_v)
            self.pos = nn.Parameter(torch.zeros(T_WIN, d_v))
            nn.init.normal_(self.pos, std=0.02)
            if layers > 0:
                layer = nn.TransformerEncoderLayer(d_v, heads, 2 * d_v, dropout=dropout, batch_first=True, activation='gelu')
                self.temporal = nn.TransformerEncoder(layer, layers)
            else:                                     # temporal ablation: the same P + e_t, a GELU, NO frame mixing
                self.temporal = None
            self.fuse = nn.Sequential(nn.Linear(d_v + d_track, d_v), nn.SiLU(), nn.Linear(d_v, d_v))
            self.head = nn.Sequential(nn.LayerNorm(2 * d_v), nn.Linear(2 * d_v, 64), nn.SiLU(), nn.Linear(64, 1))

        def window(self, xv, mv, zw):
            """xv (n, T, in_dim), mv (n, T) valid frames, zw (n, d_track) -> r_w (n, d_v)."""
            e = self.P(xv) + self.pos
            h = self.temporal(e, src_key_padding_mask=~mv) if self.temporal is not None else torch.nn.functional.gelu(e)
            m = mv[..., None].float()
            v = (h * m).sum(1) / m.sum(1).clamp(min=1)
            return self.fuse(torch.cat([v, zw], -1))

        def route(self, r, wb, ww, B, nW):
            """Scatter the window vectors into (B, nW, d_v), pool [mean, max] over each route's windows, head."""
            buf = torch.zeros(B, nW, r.shape[-1], device=r.device, dtype=r.dtype)
            bm = torch.zeros(B, nW, dtype=torch.bool, device=r.device)
            buf[wb, ww] = r
            bm[wb, ww] = True
            if getattr(self, 'capture_buf', False) and buf.requires_grad:   # --dump-attn: window saliency of this arm
                buf.retain_grad(); self._buf = buf
            m = bm[..., None].float()
            mean = (buf * m).sum(1) / m.sum(1).clamp(min=1)
            mx = buf.masked_fill(~bm[..., None], -1e9).max(1).values
            return self.head(torch.cat([mean, mx], -1)).squeeze(-1)

    return VisualWindow()


def build_window_fusion(torch, nn, vis_dim, tokens=('vis', 'track'), d=64, smart_dim=256, heads=4, dropout=0.1):
    """Token-fusion window encoder (the `--fuse-window` arm, 2026-09-12):
        v_w = mean_t MLP_v(z_t)               the window's frames -> per-frame MLP (vis_dim -> d -> d) -> mean over valid frames
        m_w = z_w                              the R2-noLane window encoder's track feature  ('track')
        s_w = MLP_s(SMART window feature)      the frozen SMART/CAT-K window feature -> d    ('smart')
        tokens = [v_w + e_vis, m_w + e_track, s_w + e_smart]   (modality embeddings; only the requested tokens)
        h = FusionTransformer(tokens)          1 layer, `heads` heads, FFN 2d, dropout
        r_w = mean over the tokens             the fused 64-d window feature, then the harness's route pooling."""
    class WindowFusion(nn.Module):
        def __init__(self):
            super().__init__()
            self.tokens = tuple(tokens)
            self.mlp_v = nn.Sequential(nn.Linear(vis_dim, d), nn.SiLU(), nn.Linear(d, d), nn.SiLU()) if 'vis' in tokens else None
            self.mlp_s = nn.Sequential(nn.Linear(smart_dim, d), nn.SiLU(), nn.Linear(d, d), nn.SiLU()) if 'smart' in tokens else None
            self.mod = nn.Parameter(torch.zeros(len(tokens), d))
            nn.init.normal_(self.mod, std=0.02)
            self.fusion = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=dropout, batch_first=True, activation='gelu'), 1)

        def forward(self, xv=None, mv=None, zw=None, xs=None):
            toks = []
            for k, name in enumerate(self.tokens):
                if name == 'vis':
                    m = mv[..., None].float()
                    t = (self.mlp_v(xv) * m).sum(1) / m.sum(1).clamp(min=1)
                elif name == 'track':
                    t = zw
                else:
                    t = self.mlp_s(xs)
                toks.append(t + self.mod[k])
            h = self.fusion(torch.stack(toks, 1))                     # (n, n_tokens, d)
            return h.mean(1)

    return WindowFusion()

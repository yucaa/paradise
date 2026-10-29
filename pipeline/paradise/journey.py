"""Run the brain on the hero fly's rendered point-of-view sequence.

Each frame is scored exactly as heaven was extracted: the brain's settled
response to that view, starting from rest. (Carrying state across frames is
not trustworthy here: near its critical gain the model has very slow modes,
so the state would depend on flight history, not on what the fly sees.)
Results are then smoothed in time for display. Writes, per frame:
  activity/####.png  retinotopic map of visual-neuron firing (drives the glow)
  journey.json       valence populations, arousal, region activity
  detail.npz         per-cell-type activity and per-neuron valence activity
                     (for the viewer's rose-window and choir modes)
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch
from PIL import Image

from .brain import Brain
from .connectome import DERIVED, ROOT, neurons
from .heaven import H, W, Eyes, calibrated_valence
from .regions import assign

BATCH = 24
SMOOTH_S = 0.3        # display smoothing time constant, seconds
FPS = 24
SPLAT_SIGMA_PX = 1.5


def load_frame(path) -> torch.Tensor:
    a = np.asarray(Image.open(path).convert("RGB").resize((W, H), Image.BILINEAR), np.float32) / 255
    return torch.tensor(a).permute(2, 0, 1).cuda()


class Splatter:
    """Columnar neurons -> (H, W) map of their positive deviation from rest."""

    def __init__(self):
        c = pd.read_parquet(DERIVED / "columns.parquet")
        c = c[c.el.between(-90, 90)]
        x = ((c.az.values + 180) / 360 * W).astype(int) % W
        y = ((90 - c.el.values) / 180 * H).clip(0, H - 1).astype(int)
        self.idx = torch.tensor(c.idx.values).cuda()
        self.cell = torch.tensor(y * W + x).cuda()
        self.count = torch.zeros(H * W).cuda().index_add_(0, self.cell, torch.ones(len(c)).cuda())
        k = torch.arange(-4, 5, dtype=torch.float32)
        k = torch.exp(-k ** 2 / (2 * SPLAT_SIGMA_PX ** 2))
        self.k = (k / k.sum()).cuda()

    def __call__(self, d: torch.Tensor) -> torch.Tensor:
        s = torch.zeros(H * W).cuda().index_add_(0, self.cell, d[self.idx].clamp(min=0))
        m = (s / self.count.clamp(min=1)).view(1, 1, H, W)
        F = torch.nn.functional
        m = F.conv2d(F.pad(m, (4, 4, 0, 0), mode="circular"), self.k.view(1, 1, 1, -1))
        m = F.conv2d(F.pad(m, (0, 0, 4, 4), mode="replicate"), self.k.view(1, 1, -1, 1))
        return m[0, 0]


def cell_types(regions, per_region=18, min_count=4):
    """The largest cell types in each region: rose-window segments."""
    n = neurons()
    t = n.cell_type.fillna("")
    names, region, idx = [], [], []
    for reg in sorted(regions.unique()):
        m = (regions.values == reg) & (t.values != "")
        counts = t[m].value_counts()
        for ty, c in counts.head(per_region).items():
            if c < min_count:
                continue
            names.append(ty)
            region.append(reg)
            idx.append(torch.tensor(np.flatnonzero(m & (t.values == ty))).cuda())
    return names, region, idx


def ema(x):
    a = 1 - np.exp(-1 / (SMOOTH_S * FPS))
    y = x.copy()
    for i in range(1, len(y)):
        y[i] = y[i - 1] + a * (x[i] - y[i - 1])
    return y


def smooth(records):
    """Exponential smoothing of every numeric field, for display. The raw,
    unsmoothed per-frame values are kept under 'raw'."""
    a = 1 - np.exp(-1 / (SMOOTH_S * FPS))
    out, prev = [], None
    for rec in records:
        cur = {k: v for k, v in rec.items() if k != "frame"}
        if prev is not None:
            cur = {k: ({r: prev[k][r] + a * (v[r] - prev[k][r]) for r in v} if isinstance(v, dict)
                       else prev[k] + a * (v - prev[k])) for k, v in cur.items()}
        out.append({"frame": rec["frame"], **cur,
                    "raw": {k: v for k, v in rec.items() if k not in ("frame", "regions")}})
        prev = cur
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default=str(ROOT / "out" / "flyview"))
    ap.add_argument("--out", default=str(ROOT / "out" / "journey"))
    ap.add_argument("--gain", type=float, default=1.3)
    args = ap.parse_args()
    from pathlib import Path
    frames = sorted(Path(args.frames).glob("*.png"))
    out = Path(args.out)
    (out / "activity").mkdir(parents=True, exist_ok=True)

    brain = Brain(gain=args.gain, leak=0.2, steps=150)
    eyes = Eyes()
    valence, _ = calibrated_valence(brain, eyes)
    regions = assign()
    names = sorted(regions.unique())
    reg_idx = {k: torch.tensor(np.flatnonzero(regions.values == k)).cuda() for k in names}
    splat = Splatter()
    types, type_region, type_idx = cell_types(regions)
    pops = valence.pops
    choir_idx = torch.tensor(np.concatenate([p.idx for p in pops])).cuda()
    choir_pop = np.concatenate([[i] * len(p.idx) for i, p in enumerate(pops)])

    maps, records, type_act, choir = [], [], [], []
    for b0 in range(0, len(frames), BATCH):
        chunk = frames[b0:b0 + BATCH]
        lum = torch.cat([eyes(load_frame(p)) for p in chunk], 1)
        r = brain(brain.eyes(lum), r0=brain.rest)
        D = r - brain.rest
        v = valence(r)
        for j in range(len(chunk)):
            d = D[:, j]
            records.append({
                "frame": b0 + j,
                **{k: float(x[j]) for k, x in v.items()},
                "regions": {k: float(d[ix].abs().mean()) for k, ix in reg_idx.items()},
                "activity": float(d.pow(2).mean()),
            })
            maps.append(splat(d).cpu())
            type_act.append([float(d[ix].abs().mean()) for ix in type_idx])
            choir.append(d[choir_idx].cpu().numpy())
        print(b0, flush=True)
    records = smooth(records)
    np.savez_compressed(out / "detail.npz", type_act=ema(np.array(type_act, np.float32)),
                        type_names=np.array(types), type_region=np.array(type_region),
                        choir=ema(np.array(choir, np.float32)), choir_pop=choir_pop,
                        pop_names=np.array([p.name for p in pops]),
                        pop_weights=np.array([p.weight for p in pops]))
    M = torch.stack(maps)
    a = 1 - np.exp(-1 / (SMOOTH_S * FPS))
    for i in range(1, len(M)):
        M[i] = M[i - 1] + a * (M[i] - M[i - 1])

    # One normalisation for the whole film so glow is comparable across frames.
    hi = torch.quantile(M.flatten()[::7], 0.995).item()
    for f, m in enumerate(M):
        a = (m / hi).clamp(0, 1).numpy()
        Image.fromarray((a * 65535).astype(np.uint16)).save(out / "activity" / f"{f:04d}.png")
    (out / "journey.json").write_text(json.dumps({"gain": args.gain, "smooth_s": SMOOTH_S,
                                                   "regions": names, "frames": records}))
    print("done", len(frames), "frames; glow scale", hi)


if __name__ == "__main__":
    main()

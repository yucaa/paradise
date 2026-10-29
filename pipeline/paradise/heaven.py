"""Extract heaven: optimise a full-sphere panorama until the fly's brain,
seeing it through its own photoreceptors, reports maximal valence.

  panorama (1/f Fourier params) -> per-photoreceptor sampling with spectral
  sensitivity -> whole-brain rate model -> valence populations -> loss
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from .brain import Brain
from .connectome import DERIVED, ROOT
from .valence import Valence

H, W = 64, 128  # equirectangular, ~2.8 deg/px (fly inter-ommatidial angle ~5 deg)

# Approximate spectral weights on (R, G, B) per photoreceptor class. Flies are
# nearly red-blind; R7 is UV-sensitive and we let blue stand in for UV.
SPECTRAL = {"R1-6": (0.10, 0.50, 0.40), "R7": (0.0, 0.05, 0.95), "R8": (0.25, 0.60, 0.15)}

# Decorrelated colour basis (sqrt of a natural-image RGB covariance), as used in
# feature-visualisation work, so steps in parameter space look like natural colour.
COLOR = torch.tensor([[0.26, 0.09, 0.02], [0.27, 0.00, -0.05], [0.27, -0.09, 0.03]])


class Panorama(torch.nn.Module):
    def __init__(self, decay: float = 1.0, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        fy = torch.fft.fftfreq(H)[:, None]
        fx = torch.fft.rfftfreq(W)[None, :]
        f = torch.sqrt(fx ** 2 + fy ** 2).clamp(min=1.0 / max(H, W))
        self.register_buffer("scale", (1.0 / f ** decay) / (H * W) ** 0.5)
        self.spec = torch.nn.Parameter(0.01 * torch.randn(3, H, W // 2 + 1, 2, generator=g))

    def forward(self) -> torch.Tensor:
        """(3, H, W) RGB in [0, 1]."""
        x = torch.fft.irfft2(torch.view_as_complex(self.spec) * self.scale, s=(H, W), norm="ortho")
        x = torch.einsum("ij,jhw->ihw", COLOR.to(x.device) / COLOR.norm(dim=0).max(), x)
        return torch.sigmoid(4 * x)


GREY_FILL = 0.5
COVER_DEG = 8.0


def _covered(r: pd.DataFrame) -> np.ndarray:
    """True where a photoreceptor has an R1-6 neighbour within COVER_DEG."""
    from scipy.spatial import cKDTree
    out = np.ones(len(r), dtype=bool)
    for side in ("left", "right"):
        m = (r.side == side).values
        lam = m & (r.cell_type == "R1-6").values
        d, _ = cKDTree(r[["az", "el"]].values[lam]).query(r[["az", "el"]].values[m])
        out[m] = d < COVER_DEG
    return out


ACCEPTANCE_DEG = 5.0  # ommatidial acceptance angle (FWHM)


def _gaussian_kernel(fwhm_px: float) -> torch.Tensor:
    sigma = fwhm_px / 2.355
    x = torch.arange(-3, 4, dtype=torch.float32)
    k = torch.exp(-x ** 2 / (2 * sigma ** 2))
    return k / k.sum()


class Eyes(torch.nn.Module):
    """Sample the panorama where each photoreceptor looks, through the
    ommatidium's ~5 degree acceptance angle."""

    def __init__(self, device="cuda"):
        super().__init__()
        self.register_buffer("kernel", _gaussian_kernel(ACCEPTANCE_DEG / (360 / W)).to(device))
        r = pd.read_parquet(DERIVED / "retina.parquet")
        az = torch.tensor(r.az.values, dtype=torch.float32)
        el = torch.tensor(r.el.values, dtype=torch.float32).clamp(-89, 89)
        self.register_buffer("grid", torch.stack([az / 180, -el / 90], -1)[None, None].to(device))
        spec = torch.tensor([SPECTRAL[t] for t in r.cell_type], dtype=torch.float32)
        # FlyWire's left lamina is incomplete: some of the field has R7/R8 but no
        # R1-6. Hold photoreceptors there at grey so the optimiser can't exploit
        # the gap in the reconstruction.
        spec *= torch.tensor(_covered(r), dtype=torch.float32)[:, None]
        self.register_buffer("spec", spec.T.to(device))  # (3, P)
        self.register_buffer("fill", torch.tensor(GREY_FILL * (spec.sum(1) == 0).numpy(),
                                                  dtype=torch.float32, device=device)[:, None])

    def forward(self, pano: torch.Tensor, yaw_px: int = 0, pitch_px: int = 0) -> torch.Tensor:
        """pano (3, H, W) -> photoreceptor luminance (P, 1). Yaw wraps."""
        p = torch.roll(pano, shifts=(pitch_px, yaw_px), dims=(1, 2))
        k = self.kernel
        p = F.conv2d(F.pad(p[:, None], (3, 3, 0, 0), mode="circular"), k.view(1, 1, 1, -1))
        p = F.conv2d(F.pad(p, (0, 0, 3, 3), mode="replicate"), k.view(1, 1, -1, 1))[:, 0]
        p = torch.cat([p[:, :, -1:], p, p[:, :, :1]], 2)  # pad for wrap at +-180
        g = self.grid.clone()
        g[..., 0] = g[..., 0] * W / (W + 2)
        s = F.grid_sample(p[None], g, mode="bilinear", align_corners=False)[0, :, 0]  # (3, P)
        return (s * self.spec).sum(0)[:, None] + self.fill


def _randomise(m):
    """Give a Panorama full-strength random 1/f content (a 'natural scene')."""
    if isinstance(m, Panorama):
        m.spec.data.normal_(0, 1.0)


@torch.no_grad()
def calibrated_valence(brain: Brain, eyes: Eyes, n_scenes: int = 32):
    """Valence readout normalised on random natural-statistics scenes, plus the
    mean squared whole-brain activity those scenes evoke."""
    valence = Valence(brain.rest)
    scenes = torch.cat([eyes(Panorama(seed=1000 + i).cuda().apply(_randomise)())
                        for i in range(n_scenes)], 1)
    rs = brain(brain.eyes(scenes), r0=brain.rest)
    valence.calibrate(rs)
    return valence, (rs - brain.rest).pow(2).mean().item()


def save(pano: torch.Tensor, path: Path):
    a = (pano.detach().permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(a).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="heaven")
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--batch", type=int, default=4, help="head jitters per step")
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--calm", type=float, default=0.1, help="weight on arousal^2 penalty")
    ap.add_argument("--budget", type=float, default=1.0,
                    help="weight on whole-brain activity, in units of a random scene's activity")
    ap.add_argument("--gain", type=float, default=1.3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = ROOT / "out" / args.name
    (out / "frames").mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    brain = Brain(gain=args.gain, leak=0.2, steps=150)
    for p in brain.parameters():
        p.requires_grad_(False)
    eyes = Eyes()
    valence, act_scale = calibrated_valence(brain, eyes)
    print("arousal mode:", dict(zip([p.name for p in valence.pops], valence.mode.round(decimals=2).tolist())))
    pano = Panorama(seed=args.seed).cuda()
    opt = torch.optim.Adam(pano.parameters(), lr=args.lr)
    log = []
    for it in range(args.iters + 1):
        img = pano()
        lum = torch.cat([eyes(img, int(np.random.randint(-6, 7)), int(np.random.randint(-3, 4)))
                         for _ in range(args.batch)], 1)
        r = brain(brain.eyes(lum), r0=brain.rest)
        v = valence(r)
        activity = (r - brain.rest).pow(2).mean(0) / act_scale
        loss = (-v["valence"] + args.calm * v["arousal"] ** 2 + args.budget * activity).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        rec = {"it": it, "loss": loss.item(), "activity": activity.mean().item(),
               **{k: x.mean().item() for k, x in v.items()}}
        log.append(rec)
        if it % 10 == 0:
            save(img, out / "frames" / f"{it:04d}.png")
            print(" ".join(f"{k}={x:.4g}" if isinstance(x, float) else f"{k}={x}" for k, x in rec.items()),
                  flush=True)
    save(pano(), out / "panorama.png")
    torch.save(pano.state_dict(), out / "panorama.pt")
    (out / "log.json").write_text(json.dumps({"args": vars(args), "log": log}))


if __name__ == "__main__":
    main()

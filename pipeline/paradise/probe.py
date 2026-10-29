"""Sweep model gain; report resting activity and visual sensitivity."""
import sys

import numpy as np
import pandas as pd
import torch

from .brain import Brain
from .connectome import DERIVED, neurons

n = neurons()
r = pd.read_parquet(DERIVED / "retina.parquet")
el = torch.tensor(r.el.values, device="cuda", dtype=torch.float32)[:, None]
az = torch.tensor(r.az.values, device="cuda", dtype=torch.float32)[:, None]
groups = {ty: torch.tensor(np.flatnonzero(n.cell_type.fillna("").str.startswith(ty).values), device="cuda")
          for ty in ["PAM", "PPL1", "MBON", "DNp01", "IPC", "LPLC2", "LC4", "KC", "Mi1", "T4"]}
for gain in map(float, sys.argv[1].split(",")):
    b = Brain(gain=gain)
    grey = b.rest[:, 0]

    def run(lum):
        return b(b.eyes(lum), r0=b.rest)[:, 0]
    stim = {"sky": run(torch.where(el > 0, 0.9, 0.1)),
            "spots": run(0.5 + 0.4 * torch.sin(az / 8) * torch.sin(el / 8))}
    print(f"g={gain}: rest active {(grey > 0.05).float().mean().item():.2f} "
          f"saturated {(grey > 0.9).float().mean().item():.2f} "
          f"converged {(b(b.eyes(torch.full_like(el, .5)), r0=b.rest)[:, 0] - grey).abs().max().item():.1e}")
    for k, ix in groups.items():
        print(f"   {k:6s} rest {grey[ix].mean().item():.3f}  " + "  ".join(
            f"{s} d {(v[ix] - grey[ix]).abs().mean().item():.3f}" for s, v in stim.items()))

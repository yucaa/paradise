"""The brain's verdict: signed neuron populations whose activity, relative to
rest, defines how much the fly 'wants to stay'.

Valence assignments follow the fly literature:
  PAM dopamine neurons signal reward; PPL1 dopamine neurons signal punishment.
  Mushroom-body output neurons: glutamatergic ones drive avoidance,
  cholinergic and GABAergic ones drive approach (Aso et al. 2014 heuristic).
  Insulin-producing cells signal a fed, sated state.
  The giant fiber (DNp01) and the looming detectors feeding it (LPLC2, LC4)
  are the escape circuit.
"""
from dataclasses import dataclass

import numpy as np
import torch

from .connectome import neurons


CAP = 3.0


@dataclass(frozen=True)
class Population:
    name: str
    weight: float
    idx: np.ndarray


def populations() -> list[Population]:
    n = neurons()
    t = n.cell_type.fillna("")
    mbon = t.str.startswith("MBON")
    pick = lambda m: np.flatnonzero(m.values)
    return [
        Population("reward", +1.0, pick(t.str.startswith("PAM"))),
        Population("approach", +0.5, pick(mbon & n.top_nt.isin(["acetylcholine", "gaba"]))),
        Population("satiety", +0.5, pick(t == "IPC")),
        Population("punishment", -1.0, pick(t.str.startswith("PPL1"))),
        Population("avoidance", -0.5, pick(mbon & (n.top_nt == "glutamate"))),
        Population("escape", -1.0, pick(t.isin(["DNp01", "LPLC2", "LC4"]))),
    ]


class Valence(torch.nn.Module):
    """Each population's mean change from rest, divided by how much that
    population varies across random natural-statistics scenes. Without this,
    populations near the eye (the looming detectors) would drown out the
    reward system several synapses deeper.

    Vision drives all of these populations together along one shared mode
    (reward and punishment correlate ~0.98 across random scenes). We call that
    mode 'arousal' and measure valence orthogonally to it, so heaven means
    good, not merely stimulating."""

    def __init__(self, rest: torch.Tensor):
        super().__init__()
        self.pops = populations()
        self.rest = rest
        for p in self.pops:
            self.register_buffer(f"ix_{p.name}", torch.tensor(p.idx, device=rest.device))
        self.register_buffer("scale", torch.ones(len(self.pops), device=rest.device))
        self.register_buffer("mode", torch.zeros(len(self.pops), device=rest.device))

    def raw(self, r: torch.Tensor) -> torch.Tensor:
        """(P, B) mean change from rest per population."""
        d = r - self.rest
        return torch.stack([d[getattr(self, f"ix_{p.name}")].mean(0) for p in self.pops])

    @torch.no_grad()
    def calibrate(self, responses: torch.Tensor):
        """responses: (N, B) rates for a batch of random scenes."""
        raw = self.raw(responses)
        self.scale = raw.std(1).clamp(min=1e-12)
        z = raw / self.scale[:, None]
        u = torch.linalg.svd(z - z.mean(1, keepdim=True), full_matrices=False).U[:, 0]
        self.mode = u * torch.sign(u.sum())

    def forward(self, r: torch.Tensor) -> dict[str, torch.Tensor]:
        """r: (N, B) rates. Returns per-population normalised change (B,),
        'arousal' (projection on the shared mode) and 'valence' (weighted sum
        of the arousal-free residuals)."""
        z = self.raw(r) / self.scale[:, None]
        arousal = self.mode @ z
        resid = z - self.mode[:, None] * arousal
        w = torch.tensor([p.weight for p in self.pops], device=z.device)
        out = {p.name: z[i] for i, p in enumerate(self.pops)}
        out["arousal"] = arousal
        # Soft-cap each population at a few sigma so no single one carries heaven.
        out["valence"] = w @ (CAP * torch.tanh(resid / CAP))
        return out

"""Differentiable whole-brain rate model on the FlyWire connectome.

r <- (1 - a) r + a * f(g * W r + bias + input),   f(x) = tanh(relu(x))

W holds signed synapse counts (post x pre), each row divided by the neuron's
total input synapses, so drive reflects the fraction of inputs that are active.
Biases are set homeostatically: with the eyes looking at neutral grey, every
neuron settles at the same set point, inside its responsive range.

This is a coarse abstraction: no neuromodulation, no plasticity, graded rates
instead of spikes. It keeps what the wiring itself implies, which is exactly
what we want to interrogate.
"""
import numpy as np
import pandas as pd
import torch

from .connectome import DERIVED, edges, neurons

GREY = 0.5


class Brain(torch.nn.Module):
    def __init__(self, gain: float = 3.0, set_point: float = 0.3, leak: float = 0.5,
                 steps: int = 60, device: str = "cuda"):
        super().__init__()
        e = edges()
        self.n = len(neurons())
        in_total = np.bincount(e.post.values, weights=e["count"].values, minlength=self.n)
        val = e["count"].values * e.sign.values * gain / in_total[e.post.values]
        idx = torch.tensor(np.stack([e.post.values, e.pre.values]))
        W = torch.sparse_coo_tensor(idx, torch.tensor(val, dtype=torch.float32),
                                    (self.n, self.n), check_invariants=False).coalesce()
        self.register_buffer("W", W.to_sparse_csr().to(device))
        self.leak, self.steps, self.device = leak, steps, device
        retina = pd.read_parquet(DERIVED / "retina.parquet")
        self.register_buffer("photo", torch.tensor(retina.idx.values, device=device))
        self.register_buffer("bias", torch.zeros(self.n, 1, device=device))
        self._calibrate(set_point)

    def eyes(self, lum: torch.Tensor) -> torch.Tensor:
        """Photoreceptor luminance (P, B) in [0, 1] -> external drive (N, B)."""
        d = torch.zeros(self.n, lum.shape[1], device=self.device, dtype=lum.dtype)
        return d.index_copy(0, self.photo, lum)

    @torch.no_grad()
    def _calibrate(self, set_point: float, rounds: int = 40):
        """Homeostasis: nudge each bias until pre-activation at grey equals
        set_point. Photoreceptors are pure inputs and keep zero bias."""
        drive = self.eyes(torch.full((len(self.photo), 1), GREY, device=self.device))
        free = torch.ones(self.n, 1, dtype=torch.bool, device=self.device)
        free[self.photo] = False
        r = torch.zeros_like(drive)
        for _ in range(rounds):
            r = self.forward(drive, r0=r)
            x = torch.sparse.mm(self.W, r) + self.bias + drive
            self.bias += torch.where(free, set_point - x, 0.0)
        self.rest = self.forward(drive, r0=r)

    def forward(self, drive: torch.Tensor, steps: int | None = None,
                r0: torch.Tensor | None = None, record: bool = False):
        """drive: (N, B) external input, held constant. Returns final rates
        (N, B) and, if record, the full trajectory (steps, N, B)."""
        r = torch.zeros_like(drive) if r0 is None else r0.expand_as(drive)
        traj = []
        for _ in range(steps or self.steps):
            x = torch.sparse.mm(self.W, r) + self.bias + drive
            r = (1 - self.leak) * r + self.leak * torch.tanh(torch.relu(x))
            if record:
                traj.append(r)
        return (r, torch.stack(traj)) if record else r

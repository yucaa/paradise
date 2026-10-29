"""Live brain server for the real-time Paradise (viewer/live/).

The browser renders what the hero fly sees (128x64 equirectangular, body
frame, raw panorama values) and sends it here over a websocket. Each view is
scored exactly as heaven was extracted: the brain's settled response from rest.
The reply carries:
  text   {valence readout, region activity}
  binary int8[N]  per-neuron activity of the constellation sample, in units of
                  that neuron's own variability across natural scenes (x40)
         uint8[64*128] retinotopic glow map (drives the meadow's light)

`python -m paradise.live export` writes the constellation sample and its
strongest synapses to viewer/live/data/constellation.json.
`python -m paradise.live serve` runs the server on ws://127.0.0.1:8766.
"""
import asyncio
import base64
import json
import shutil
import sys
import time

import numpy as np
import torch

from .brain import Brain
from .connectome import ROOT, edges, neurons
from .heaven import H, W, Eyes, Panorama, _randomise, calibrated_valence
from .journey import Splatter
from .regions import assign

DATA = ROOT / "viewer" / "live" / "data"
SAMPLE = 36000
N_EDGES = 26000
PORT = 8766


def sample_ids():
    """Constellation neurons: every valence neuron plus a random draw."""
    path = DATA / "sample.npy"
    if path.exists():
        return np.load(path)
    from .valence import populations
    n = neurons()
    ok = np.flatnonzero(n[["x", "y", "z"]].notna().all(axis=1).values)
    judges = np.concatenate([p.idx for p in populations()])
    rest = np.setdiff1d(ok, judges)
    rng = np.random.default_rng(0)
    ids = np.sort(np.concatenate([judges, rng.choice(rest, SAMPLE - len(judges), replace=False)]))
    DATA.mkdir(parents=True, exist_ok=True)
    np.save(path, ids)
    return ids


def export():
    ids = sample_ids()
    n, reg = neurons(), assign()
    names = sorted(reg.unique())
    P = n[["x", "y", "z"]].values[ids].astype(np.float32)
    P -= (P.min(0) + P.max(0)) / 2
    P /= np.abs(P).max()
    P[:, 1] *= -1                                    # dorsal up
    from .valence import populations
    judge = np.full(len(ids), -1, np.int8)
    pos_in = {g: i for i, g in enumerate(ids)}
    for k, p in enumerate(populations()):
        for g in p.idx:
            judge[pos_in[g]] = k
    e = edges()
    m = e.pre.isin(pos_in) & e.post.isin(pos_in) & (e.pre != e.post)
    s = e[m].nlargest(N_EDGES, "count")
    pairs = np.c_[s.pre.map(pos_in).values, s.post.map(pos_in).values].astype(np.uint32)
    b64 = lambda a: base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()
    (DATA / "constellation.json").write_text(json.dumps({
        "n": len(ids), "regions": names,
        "pos": b64(P), "region": b64(np.array([names.index(r) for r in reg.values[ids]], np.uint8)),
        "judge": b64(judge), "judge_names": [p.name for p in populations()],
        "edges": b64(pairs), "edge_sign": b64(s.sign.values.astype(np.int8)),
        "edge_w": b64(np.clip(np.log2(s["count"].values), 0, 255).astype(np.uint8))}))
    print("constellation:", len(ids), "neurons,", len(pairs), "synapses")
    shutil.copy(ROOT / "out" / "heaven_current.png", DATA / "heaven.png")   # the extracted heaven, for the meadow


class LiveBrain:
    def __init__(self):
        t = time.time()
        self.brain = Brain(gain=1.3, leak=0.2, steps=150)
        self.eyes = Eyes()
        self.valence, _ = calibrated_valence(self.brain, self.eyes)
        self.splat = Splatter()
        self.ids = torch.tensor(sample_ids()).cuda()
        reg = assign()
        self.region_names = sorted(reg.unique())
        self.reg_idx = [torch.tensor(np.flatnonzero(reg.values == r)).cuda() for r in self.region_names]
        with torch.no_grad():
            scenes = torch.cat([self.eyes(Panorama(seed=2000 + i).cuda().apply(_randomise)())
                                for i in range(32)], 1)
            D = self.brain(self.brain.eyes(scenes), r0=self.brain.rest) - self.brain.rest
            sd = D[self.ids].std(1)
            self.scale = sd.clamp(min=float(sd.median()) * 0.05)
            maps = torch.stack([self.splat(D[:, j]) for j in range(D.shape[1])])
            self.glow_scale = float(torch.quantile(maps.flatten()[::5], 0.995))
        print(f"live brain ready in {time.time() - t:.1f}s", flush=True)

    @torch.no_grad()
    def step(self, rgb: bytes):
        pano = torch.frombuffer(bytearray(rgb), dtype=torch.uint8).view(H, W, 3).cuda()
        pano = pano.permute(2, 0, 1).float() / 255
        r = self.brain(self.brain.eyes(self.eyes(pano)), r0=self.brain.rest)
        d = (r - self.brain.rest)[:, 0]
        v = self.valence(r)
        z = (d[self.ids] / self.scale * 40).clamp(-127, 127).to(torch.int8).cpu().numpy()
        glow = (self.splat(d) / self.glow_scale * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
        meta = {k: round(float(x), 4) for k, x in v.items()}
        meta["regions"] = {k: float(d[ix].abs().mean()) for k, ix in zip(self.region_names, self.reg_idx)}
        return meta, z.tobytes() + glow.tobytes()


async def serve():
    import websockets
    live = LiveBrain()

    async def handler(ws):
        try:
            await respond(ws)
        except websockets.ConnectionClosed:
            pass                         # viewer closed or reloaded mid-reply

    async def respond(ws):
        first = True
        async for msg in ws:
            if isinstance(msg, (bytes, bytearray)) and len(msg) == H * W * 3:
                if first:   # keep the first view of each connection, for checking the eye
                    from PIL import Image
                    Image.frombytes("RGB", (W, H), bytes(msg)).save(ROOT / "out" / "live_eye.png")
                    first = False
                t = time.perf_counter()
                meta, blob = live.step(msg)
                meta["ms"] = round((time.perf_counter() - t) * 1000, 1)
                await ws.send(json.dumps(meta))
                await ws.send(blob)

    async with websockets.serve(handler, "127.0.0.1", PORT, max_size=2 ** 22):
        print(f"serving ws://127.0.0.1:{PORT}", flush=True)
        await asyncio.Future()


if __name__ == "__main__":
    {"export": export, "serve": lambda: asyncio.run(serve())}[sys.argv[1]]()

"""Pack simulation results for the local viewer (viewer/index.html).

  viewer/data/journey.json   per-frame valence readout
  viewer/data/brain.json     soma point cloud (normalised) with region ids
  viewer/data/detail.json    cell-type activity (rose window), valence-neuron
                             activity (choir), quantised and base64-packed
"""
import base64
import json

import numpy as np

from .connectome import ROOT, neurons
from .regions import assign

OUT = ROOT / "viewer" / "data"
J = ROOT / "out" / "journey"

# Rose-window rings, from the eye inward.
RING_ORDER = ["retina", "optic lobe", "visual projection", "central brain", "mushroom body",
              "central complex", "lateral horn", "antennal lobe", "sensory", "neuroendocrine", "descending"]


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    jr = json.loads((J / "journey.json").read_text())
    keep = ["valence", "arousal", "reward", "approach", "satiety", "punishment", "avoidance", "escape"]
    (OUT / "journey.json").write_text(json.dumps({
        "fps": 24, "regions": jr["regions"],
        "frames": [{**{k: round(f[k], 4) for k in keep},
                    "regions": {k: v for k, v in f["regions"].items()}} for f in jr["frames"]]}))

    n = neurons()
    reg = assign()
    names = sorted(reg.unique())
    ok = n[["x", "y", "z"]].notna().all(axis=1).values
    rng = np.random.default_rng(0)
    idx = rng.choice(np.flatnonzero(ok), 36000, replace=False)
    P = n[["x", "y", "z"]].values[idx].astype(np.float32)
    P -= (P.min(0) + P.max(0)) / 2
    P /= np.abs(P).max()
    P[:, 1] *= -1                       # brain y grows ventrally; flip so dorsal is up
    rid = np.array([names.index(r) for r in reg.values[idx]], np.uint8)
    (OUT / "brain.json").write_text(json.dumps({"regions": names, "pos": b64(P), "region": b64(rid)}))

    d = np.load(J / "detail.npz")
    T = d["type_act"]                                     # (F, types) mean |d|
    lt = np.log10(T + 1e-9)
    lo, hi = np.percentile(lt, 2, axis=0), np.percentile(lt, 99.5, axis=0)
    # Each segment normalised to its own range over the film, plus its absolute
    # (log) level, so both "which parts of the brain are loud" and "what changes"
    # are visible.
    rel = np.clip((lt - lo) / (hi - lo + 1e-9), 0, 1)
    absl = np.clip((lt + 6) / (np.log10(3e-2) + 6), 0, 1)
    tq = (255 * (0.55 * rel + 0.45 * absl)).astype(np.uint8)
    C = d["choir"]                                        # (F, neurons) signed d
    scale = np.percentile(np.abs(C), 99, axis=0) + 1e-12
    cq = np.clip(np.round(127 * C / scale), -127, 127).astype(np.int8)
    rings = []
    for r in d["type_region"]:
        base = r.rsplit(" ", 1)[0] if r.endswith((" L", " R")) else r
        rings.append(RING_ORDER.index(base))
    (OUT / "detail.json").write_text(json.dumps({
        "frames": int(T.shape[0]),
        "types": d["type_names"].tolist(), "type_region": d["type_region"].tolist(), "type_ring": rings,
        "ring_names": RING_ORDER, "type_act": b64(tq),
        "choir": b64(cq), "choir_n": int(C.shape[1]), "choir_pop": d["choir_pop"].tolist(),
        "pop_names": d["pop_names"].tolist(), "pop_weights": d["pop_weights"].tolist()}))
    print("viewer data written:", {p.name: p.stat().st_size // 1024 for p in OUT.glob("*.json")}, "KB")


if __name__ == "__main__":
    main()

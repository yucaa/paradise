"""Recover each photoreceptor's viewing direction from optic-lobe wiring.

Columnar visual neurons wire mostly within their own and neighbouring columns,
so their graph is a 2D lattice. Flattening it (landmark Isomap) gives
retinotopic sheet coordinates, which we orient with soma/arbor positions and
map onto each eye's field of view.
"""
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

from .connectome import DERIVED, PHOTORECEPTORS, edges, neurons

SCAFFOLD = ("L1", "L2", "L3", "L5", "Mi1", "Mi4", "Mi9", "Tm1",
                             "Tm2", "Tm9", "Tm20", "C2", "C3", "T1", "Tm3")
MIN_SYN = 3
N_LANDMARKS = 600

# Approximate field of view of one (right) eye, degrees.
AZ_RANGE = (-15.0, 165.0)   # 0 = straight ahead, positive = toward the eye's side
EL_RANGE = (-65.0, 85.0)


def _embed(idx: np.ndarray, e: pd.DataFrame, rng) -> np.ndarray:
    local = {g: i for i, g in enumerate(idx)}
    m = e.pre.isin(local) & e.post.isin(local) & (e["count"] >= MIN_SYN)
    s = e[m]
    i, j = s.pre.map(local).values, s.post.map(local).values
    g = coo_matrix((np.ones(len(i)), (i, j)), shape=(len(idx),) * 2).tocsr()
    g = ((g + g.T) > 0).astype(float)
    _, lab = connected_components(g, directed=False)
    main = lab == np.bincount(lab).argmax()
    land = rng.choice(np.flatnonzero(main), N_LANDMARKS, replace=False)
    d = dijkstra(g, indices=land, unweighted=True)        # (L, n)
    dl = d[:, land]
    # Classical MDS on landmarks, then triangulate everyone else.
    J = np.eye(len(land)) - 1.0 / len(land)
    B = -0.5 * J @ (dl ** 2) @ J
    w, v = np.linalg.eigh(B)
    w, v = w[::-1][:2], v[:, ::-1][:, :2]
    pinv = v / np.sqrt(w)
    mean_sq = (dl ** 2).mean(1, keepdims=True)
    uv = np.full((len(idx), 2), np.nan)
    dm = d[:, main]
    uv[main] = (-0.5 * pinv.T @ (dm ** 2 - mean_sq)).T
    return uv


def _to_angles(uv: np.ndarray, pos: np.ndarray, side: str, lamina: np.ndarray) -> np.ndarray:
    """Rotate the sheet so its axes align with dorsal-ventral and
    anterior-posterior, then stretch to the eye's field of view.

    Dorsal-ventral order survives the optic chiasm, so every cell's y can
    orient elevation. Anterior-posterior flips at the chiasm, so only lamina
    cells (R1-6 arbors) are trusted for the azimuth sign.
    """
    u = uv - uv.mean(0)
    # Brain y grows ventrally; elevation should grow dorsally.
    coef, *_ = np.linalg.lstsq(u, -(pos[:, 1] - pos[:, 1].mean()), rcond=None)
    el_dir = coef / np.linalg.norm(coef)
    az_dir = np.array([-el_dir[1], el_dir[0]])
    el = u @ el_dir
    az = u @ az_dir
    # Anterior visual field sits at anterior lamina (small z). Flip so that
    # azimuth grows posteriorly.
    if np.corrcoef(az[lamina], pos[lamina, 2])[0, 1] < 0:
        az = -az

    def stretch(x, lo, hi):
        a, b = np.percentile(x, [1, 99])
        return lo + (x - a) / (b - a) * (hi - lo)

    az = stretch(az, *AZ_RANGE)
    el = stretch(el, *EL_RANGE)
    if side == "left":
        az = -az
    return np.c_[az, el]


def _place_photoreceptors(idx_scaf, uv_scaf, side, n, e):
    """Each photoreceptor sits at the synapse-weighted mean of its scaffold
    partners (R1-6 -> lamina cartridge, R7/R8 -> medulla column)."""
    ok = ~np.isnan(uv_scaf[:, 0])
    pos = pd.Series(np.arange(ok.sum()), index=idx_scaf[ok])
    uv = uv_scaf[ok]
    photo = np.flatnonzero(n.cell_type.isin(PHOTORECEPTORS).values & (n.side == side).values)
    s = e[e.pre.isin(photo) & e.post.isin(pos.index)]
    s = pd.concat([s, e[e.post.isin(photo) & e.pre.isin(pos.index)]
                   .rename(columns={"pre": "post", "post": "pre"})])
    k = pos[s.post.values].values
    w = s["count"].values
    num = pd.DataFrame(uv[k] * w[:, None]).groupby(s.pre.values).sum()
    den = pd.Series(w).groupby(s.pre.values).sum()
    placed = num.div(den, axis=0)
    return placed.index.values, placed.values


def build(seed: int = 0) -> pd.DataFrame:
    n, e = neurons(), edges()
    rng = np.random.default_rng(seed)
    out, cols = [], []
    for side in ("left", "right"):
        idx = np.flatnonzero(n.cell_type.isin(SCAFFOLD).values & (n.side == side).values)
        uv_scaf = _embed(idx, e, rng)
        pidx, uv = _place_photoreceptors(idx, uv_scaf, side, n, e)
        # Orient with R1-6 arbors only: they sit in the lamina, retinotopic
        # with the retina (before the first optic chiasm).
        types = n.cell_type.values[pidx]
        pos = n[["pos_x", "pos_y", "pos_z"]].values[pidx].astype(float)
        ang = _to_angles(uv, pos, side, lamina=types == "R1-6")
        out.append(pd.DataFrame({"idx": pidx, "cell_type": types,
                                 "side": side, "az": ang[:, 0], "el": ang[:, 1]}))
        # The sheet -> angle map is affine; recover it from the photoreceptors
        # and carry the columnar scaffold neurons onto the visual sphere too.
        A = np.c_[uv, np.ones(len(uv))]
        M, *_ = np.linalg.lstsq(A, ang, rcond=None)
        ok = ~np.isnan(uv_scaf[:, 0])
        sang = np.c_[uv_scaf[ok], np.ones(ok.sum())] @ M
        cols.append(pd.DataFrame({"idx": idx[ok], "cell_type": n.cell_type.values[idx[ok]],
                                  "side": side, "az": sang[:, 0], "el": sang[:, 1]}))
    df = pd.concat(out, ignore_index=True)
    DERIVED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DERIVED / "retina.parquet")
    pd.concat(cols, ignore_index=True).to_parquet(DERIVED / "columns.parquet")
    return df


if __name__ == "__main__":
    df = build()
    print(df.groupby(["side", "cell_type"]).size())
    print(df.groupby("side")[["az", "el"]].describe().T)

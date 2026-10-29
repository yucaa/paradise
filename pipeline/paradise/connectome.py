"""FlyWire v783 connectome + annotations, loaded once and cached."""
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
DERIVED = ROOT / "data" / "derived"
VOXEL_UM = np.array([4, 4, 40]) / 1000.0

PHOTORECEPTORS = ("R1-6", "R7", "R8")


@lru_cache
def edges() -> pd.DataFrame:
    """Signed synapse table: pre, post (0-based neuron index), count, sign."""
    c = pd.read_parquet(RAW / "Drosophila_brain_model" / "Connectivity_783.parquet")
    e = pd.DataFrame({
        "pre": c.Presynaptic_Index.values.astype(np.int64),
        "post": c.Postsynaptic_Index.values.astype(np.int64),
        "count": c.Connectivity.values.astype(np.float32),
        "sign": c.Excitatory.values.astype(np.float32),
    })
    # Photoreceptors are histaminergic (inhibitory); the NT classifier has no
    # histamine class and mislabels them, so force the sign here.
    n = neurons()
    is_photo = n.cell_type.isin(PHOTORECEPTORS).values
    e.loc[is_photo[e.pre.values], "sign"] = -1.0
    return e


@lru_cache
def neurons() -> pd.DataFrame:
    """One row per model neuron, indexed 0..N-1 in the connectivity's order."""
    c = pd.read_parquet(RAW / "Drosophila_brain_model" / "Connectivity_783.parquet",
                        columns=["Presynaptic_ID", "Presynaptic_Index",
                                 "Postsynaptic_ID", "Postsynaptic_Index"])
    ids = pd.concat([
        c[["Presynaptic_ID", "Presynaptic_Index"]].set_axis(["root_id", "idx"], axis=1),
        c[["Postsynaptic_ID", "Postsynaptic_Index"]].set_axis(["root_id", "idx"], axis=1),
    ]).drop_duplicates("idx").sort_values("idx")
    assert (ids.idx.values == np.arange(len(ids))).all()
    a = pd.read_csv(RAW / "flywire_annotations" / "supplemental_files"
                    / "Supplemental_file1_neuron_annotations.tsv",
                    sep="\t", low_memory=False)
    keep = ["root_id", "super_class", "cell_class", "cell_sub_class", "cell_type",
            "top_nt", "side", "pos_x", "pos_y", "pos_z", "soma_x", "soma_y", "soma_z"]
    n = ids.merge(a[keep], on="root_id", how="left").reset_index(drop=True)
    n = n.drop(columns="idx")
    xyz = n[["soma_x", "soma_y", "soma_z"]].fillna(n[["pos_x", "pos_y", "pos_z"]].set_axis(
        ["soma_x", "soma_y", "soma_z"], axis=1))
    n[["x", "y", "z"]] = xyz.values * VOXEL_UM
    return n

"""Coarse brain regions for the dashboard diagram: each neuron gets one label."""
import numpy as np
import pandas as pd

from .connectome import PHOTORECEPTORS, neurons

# Checked in order; first match wins.
RULES = [
    ("retina", lambda n: n.cell_type.isin(PHOTORECEPTORS)),
    ("optic lobe", lambda n: n.super_class.isin(["optic", "visual_centrifugal"])),
    ("visual projection", lambda n: n.super_class == "visual_projection"),
    ("mushroom body", lambda n: n.cell_class.isin(["Kenyon_Cell", "MBON", "MBIN", "DAN"])),
    ("central complex", lambda n: n.cell_class.isin(["CX", "TuBu"])),
    ("lateral horn", lambda n: n.cell_class.isin(["LHLN", "LHCENT"])),
    ("antennal lobe", lambda n: n.cell_class.isin(["ALPN", "ALLN", "ALIN", "ALON", "olfactory", "TPN"])),
    ("neuroendocrine", lambda n: (n.super_class == "endocrine")
     | n.cell_class.isin(["pars_intercerebralis", "pars_lateralis"])),
    ("descending", lambda n: n.super_class.isin(["descending", "motor"])),
    ("sensory", lambda n: n.super_class.isin(["sensory", "sensory_ascending", "ascending"])),
]


def assign() -> pd.Series:
    n = neurons()
    label = pd.Series("central brain", index=n.index)
    done = np.zeros(len(n), dtype=bool)
    for name, rule in RULES:
        m = rule(n).fillna(False).values & ~done
        label[m] = name
        done |= m
    # Visual regions are drawn per hemisphere.
    sided = label.isin(["retina", "optic lobe", "visual projection"])
    label[sided] = label[sided] + " " + n.side[sided].fillna("?").str[0].str.upper()
    return label


if __name__ == "__main__":
    print(assign().value_counts())

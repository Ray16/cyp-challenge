"""Target (CYP pocket) descriptors and ligand×pocket interaction features.

This is the piece that lets ONE model both (a) model each CYP on its own physics
and (b) generalize to an unseen target. Instead of four hardcoded per-isoform
models, we describe each pocket by a handful of PHYSICAL properties and let the
model learn ligand×pocket interaction terms. A new target enters simply as a new
row of pocket descriptors — the learned physics (charge complementarity,
hydrophobic burial, planar fit, heme coordination) transfers by construction.

Pocket descriptor values below are hand-curated from CYP structural biology
(literature pocket volumes and the key catalytic-site residues). They are meant to
be REPLACED by values computed directly from a crystal or predicted structure
(volume from a pocket-detection run; net charge / hydrophobicity / aromaticity from
the lining residues) so the same pipeline applies to any novel CYP or, more
broadly, any protein target — see compute_pocket_descriptors_from_structure (TODO).

Provenance of the hand values:
- volume (Å³): 1A2 narrow planar (~375), 2C9 (~470), 2D6 (~540), 3A4 huge (~1400).
- net_charge: the discriminating term. 2C9 pocket is CATIONIC (Arg108) → +1;
  2D6 pocket is ANIONIC (Asp301/Glu216) → -1; 1A2/3A4 ~neutral.
- planarity_pref: 1A2's narrow flat slot (Phe-rich) strongly selects planar ligands.
- hydrophobicity: 3A4/1A2 greasy; 2C9/2D6 more polar.
- heme_access: all CYPs carry the catalytic heme; accessibility for direct N
  coordination is high in the open 3A4/1A2, slightly lower in tighter pockets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# columns: volume, hydrophobicity[0-1], net_charge, planarity_pref[0-1],
#          hbond_capacity[0-1], heme_access[0-1]
POCKET_DESCRIPTORS: dict[str, dict[str, float]] = {
    "CYP1A2": dict(pocket_volume=375, pocket_hydrophobicity=0.70, pocket_net_charge=0.0,
                   pocket_planarity_pref=1.00, pocket_hbond=0.30, pocket_heme_access=0.80),
    "CYP2C9": dict(pocket_volume=470, pocket_hydrophobicity=0.60, pocket_net_charge=+1.0,
                   pocket_planarity_pref=0.20, pocket_hbond=0.60, pocket_heme_access=0.70),
    "CYP2D6": dict(pocket_volume=540, pocket_hydrophobicity=0.50, pocket_net_charge=-1.0,
                   pocket_planarity_pref=0.30, pocket_hbond=0.55, pocket_heme_access=0.70),
    "CYP3A4": dict(pocket_volume=1400, pocket_hydrophobicity=0.80, pocket_net_charge=0.0,
                   pocket_planarity_pref=0.20, pocket_hbond=0.40, pocket_heme_access=0.90),
}
POCKET_COLS = list(next(iter(POCKET_DESCRIPTORS.values())).keys())


def pocket_row(isoform: str) -> dict[str, float]:
    return POCKET_DESCRIPTORS[isoform]


def interaction_features(ligand: pd.DataFrame, pocket: dict[str, float]) -> pd.DataFrame:
    """Physically-motivated ligand×pocket cross terms (sign chosen so that larger =
    more favorable binding = higher pIC50). All transfer to a new pocket given its
    descriptors.

    Requires ligand columns from features.featurize_smiles: net_charge_pH74,
    MolLogP, FracAromaticAtoms, n_heme_coord_total, MolWt.
    """
    lg = ligand
    out = pd.DataFrame(index=lg.index)
    # salt-bridge complementarity: cation↔anionic pocket (2D6) or anion↔cationic (2C9)
    out["ix_salt_complementarity"] = -(lg["net_charge_pH74"] * pocket["pocket_net_charge"])
    # hydrophobic burial (3A4/1A2)
    out["ix_hydrophobic"] = lg["MolLogP"] * pocket["pocket_hydrophobicity"]
    # planar fit into a flat slot (1A2)
    out["ix_planar_fit"] = lg["FracAromaticAtoms"] * pocket["pocket_planarity_pref"]
    # Type II heme coordination, gated by pocket accessibility
    out["ix_heme_coord"] = lg["n_heme_coord_total"] * pocket["pocket_heme_access"]
    # steric fit: ligand heavy-atom mass relative to pocket volume (clash if too big)
    out["ix_size_ratio"] = lg["MolWt"] / pocket["pocket_volume"]
    return out


def build_pcm_matrix(ligand: pd.DataFrame, isoform: str,
                     ligand_cols: list[str]) -> pd.DataFrame:
    """One (compound, target) design matrix: ligand features ⊕ pocket descriptors ⊕
    interaction terms, for a given isoform. Stack these across isoforms to get the
    pooled proteochemometric training set."""
    pk = pocket_row(isoform)
    X = ligand[ligand_cols].copy().reset_index(drop=True)
    for c in POCKET_COLS:
        X[c] = pk[c]
    ix = interaction_features(ligand.reset_index(drop=True), pk)
    return pd.concat([X, ix.reset_index(drop=True)], axis=1)

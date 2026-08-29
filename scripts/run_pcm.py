"""Proteochemometric (PCM) model: ONE model over (ligand ⊕ pocket ⊕ interaction)
features, so each CYP is modelled through its own pocket physics yet an unseen
target is just a new descriptor row. We test generalization to a NEW TARGET with
leave-one-CYP-out, in two regimes:

  transfer    : hold out target H; train on the other 3 targets (a test compound
                MAY appear via its other-target rows). Tests pure target transfer.
  joint       : hold out target H AND remove the held-out compounds entirely from
                training. Tests the hardest case — NEW MOLECULE + NEW TARGET.

Reference points per held-out target:
  const       : predict the pooled-train mean pIC50 (a target-agnostic model's floor)
  specialized : physics model trained ON that target, scaffold-CV (upper bound —
                the number from run_ablation's physics2d)

Run (fep env):
  python scripts/run_pcm.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rdkit import Chem  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402

from cyp_challenge import config, data, metrics  # noqa: E402
from cyp_challenge.features import (  # noqa: E402
    _aromatic_planarity_block, _motif_block, _protonation_block, _thermo_block,
    featurize_smiles, murcko_scaffold,
)
from cyp_challenge.targets import build_pcm_matrix  # noqa: E402

ISOFORMS = config.ISOFORMS
_s = Chem.MolFromSmiles("c1ccccc1CCN")
LIGAND_COLS = (list(_thermo_block(_s)) + list(_aromatic_planarity_block(_s))
               + list(_motif_block(_s)) + list(_protonation_block(_s)))


def make_model():
    return HistGradientBoostingRegressor(
        loss="absolute_error", learning_rate=0.06, max_iter=400,
        max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.1, random_state=0,
    )


def build_pool(inh, feats, valid):
    """Long-format pooled PCM dataset across all isoforms."""
    scaffold = inh["SMILES"].map(murcko_scaffold).to_numpy()
    blocks = []
    for iso in ISOFORMS:
        ycol = f"{iso}_pIC50_direct_inhibition"
        has = inh[ycol].notna().to_numpy() & valid
        X = build_pcm_matrix(feats.loc[has], iso, LIGAND_COLS)
        X["_iso"] = iso
        X["_y"] = inh.loc[has, ycol].to_numpy()
        X["_lo"] = inh.loc[has, f"{ycol}_conf_low"].to_numpy()
        X["_hi"] = inh.loc[has, f"{ycol}_conf_high"].to_numpy()
        X["_cid"] = inh.loc[has, "Molecule_Name"].to_numpy()
        X["_scaf"] = scaffold[has]
        blocks.append(X.reset_index(drop=True))
    return pd.concat(blocks, ignore_index=True)


def main():
    inh = data.load_train_inhibition()
    feats = featurize_smiles(inh["SMILES"])
    valid = feats.attrs["valid"]
    pool = build_pool(inh, feats, valid)
    feat_cols = [c for c in pool.columns if not c.startswith("_")]
    print(f"Pooled PCM set: {len(pool)} (compound,target) rows, {len(feat_cols)} features")

    print("\n=== Leave-one-CYP-out: generalization to an UNSEEN target ===")
    print(f"{'target':>7} {'transfer':>9} {'joint':>7} {'const':>7}   (ST-RAE, lower=better)")
    rows = []
    for H in ISOFORMS:
        te = pool["_iso"].to_numpy() == H
        y, lo, hi = pool["_y"].to_numpy(), pool["_lo"].to_numpy(), pool["_hi"].to_numpy()

        # regime 1: transfer (compound may appear via other-target rows)
        m1 = make_model(); m1.fit(pool.loc[~te, feat_cols], y[~te])
        p1 = m1.predict(pool.loc[te, feat_cols])
        st_transfer = metrics.st_rae(y[te], p1, lo[te], hi[te])

        # regime 2: joint — remove held-out compounds from training entirely
        held_cids = set(pool.loc[te, "_cid"])
        tr2 = (~te) & (~pool["_cid"].isin(held_cids)).to_numpy()
        m2 = make_model(); m2.fit(pool.loc[tr2, feat_cols], y[tr2])
        p2 = m2.predict(pool.loc[te, feat_cols])
        st_joint = metrics.st_rae(y[te], p2, lo[te], hi[te])

        # constant floor: pooled-train mean (target-agnostic)
        const = np.full(te.sum(), y[~te].mean())
        st_const = metrics.st_rae(y[te], const, lo[te], hi[te])

        rows.append((H, st_transfer, st_joint, st_const))
        print(f"{H[3:]:>7} {st_transfer:9.3f} {st_joint:7.3f} {st_const:7.3f}")

    arr = np.array([(a, b, c) for _, a, b, c in rows])
    print(f"{'macro':>7} {arr[:,0].mean():9.3f} {arr[:,1].mean():7.3f} {arr[:,2].mean():7.3f}")
    print("\nInterpretation: transfer/joint < const ⇒ pocket descriptors carry")
    print("real, transferable signal to a target the model never trained on.")


if __name__ == "__main__":
    main()

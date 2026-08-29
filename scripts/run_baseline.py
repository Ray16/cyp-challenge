"""Tier-0/1 baseline: per-isoform gradient-boosted regressor on physics-informed
descriptors, evaluated with scaffold-disjoint CV under a soft-threshold (ST-RAE
proxy) metric, then refit on all data to produce a valid blinded-test submission.

This is the number the later physics/multi-fidelity tiers must beat. Run with the
`fep` conda env (has rdkit, sklearn, scipy, pandas):

    python scripts/run_baseline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402

from cyp_challenge import config, data, metrics  # noqa: E402
from cyp_challenge.features import featurize_smiles  # noqa: E402
from cyp_challenge.splits import scaffold_folds  # noqa: E402

ISOFORMS = config.ISOFORMS
N_SPLITS = 5
SEED = 0
CACHE = str(ROOT / "data" / "processed" / "feat_cache.pkl")


def make_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="absolute_error",       # L1 is closer to the |error| ST-RAE shape than L2
        learning_rate=0.05,
        max_iter=600,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=SEED,
    )


def main() -> None:
    inh = data.load_train_inhibition()
    test = data.load_test_blinded()

    print(f"Featurizing {len(inh)} train + {len(test)} test compounds (physics + 3D) ...")
    Xtr_all = featurize_smiles(inh["SMILES"], include_3d=True, cache_path=CACHE)
    Xte = featurize_smiles(test["SMILES"], include_3d=True, cache_path=CACHE)
    feat_cols = list(Xtr_all.columns)
    print(f"  {len(feat_cols)} features; "
          f"{(~Xtr_all.attrs['valid']).sum()} train / {(~Xte.attrs['valid']).sum()} test parse failures")

    folds = scaffold_folds(inh["SMILES"], n_splits=N_SPLITS, seed=SEED)

    test_pred = {}
    print("\n=== Scaffold-CV: model ST-RAE vs no-skill constant (lower=better) ===")
    macro_model, macro_const, macro_mae = [], [], []
    for iso in ISOFORMS:
        ycol = f"{iso}_pIC50_direct_inhibition"
        lo_col, hi_col = f"{ycol}_conf_low", f"{ycol}_conf_high"
        has = inh[ycol].notna().to_numpy() & Xtr_all.attrs["valid"]

        oof = np.full(len(inh), np.nan)
        for f in range(N_SPLITS):
            tr = has & (folds != f)
            va = has & (folds == f)
            if va.sum() == 0 or tr.sum() == 0:
                continue
            model = make_model()
            model.fit(Xtr_all.loc[tr, feat_cols], inh.loc[tr, ycol])
            oof[va] = model.predict(Xtr_all.loc[va, feat_cols])

        m = metrics.report(
            inh.loc[has, ycol], oof[has],
            inh.loc[has, lo_col], inh.loc[has, hi_col],
        )
        macro_model.append(m["st_rae"]); macro_const.append(m["const_st_rae"]); macro_mae.append(m["mae"])
        skill = (m["const_st_rae"] - m["st_rae"]) / m["const_st_rae"] * 100
        print(f"  {iso}: n={m['n']:4d}  MAE={m['mae']:.3f}  "
              f"ST-RAE={m['st_rae']:.3f}  (const={m['const_st_rae']:.3f}, "
              f"skill gain {skill:+.0f}%)")

        # refit on all available data for this isoform → test predictions
        model = make_model()
        model.fit(Xtr_all.loc[has, feat_cols], inh.loc[has, ycol])
        test_pred[ycol] = model.predict(Xte[feat_cols])

    print(f"\n  MA-ST-RAE (model)    = {np.mean(macro_model):.4f}")
    print(f"  MA-ST-RAE (constant) = {np.mean(macro_const):.4f}   <- no-skill floor")
    print(f"  MA-MAE               = {np.mean(macro_mae):.4f}")

    # assemble submission
    sub = test[["SMILES", "Molecule_Name"]].copy()
    for ycol in [f"{iso}_pIC50_direct_inhibition" for iso in ISOFORMS]:
        sub[ycol] = test_pred[ycol]
    out = ROOT / "submissions" / "regression" / "physics3d_perisoform.csv"
    sub.to_csv(out, index=False)
    print(f"\nWrote submission → {out.relative_to(ROOT)}  ({len(sub)} rows)")


if __name__ == "__main__":
    main()

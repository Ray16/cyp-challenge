"""Ablation: does physics-informed featurization beat the vibe-coder baseline,
and specifically do the protonation block (→ CYP2D6) and 3D planarity block
(→ CYP1A2) close the gaps the per-isoform diagnostic pointed at?

Feature sets, all through the same scaffold-CV + ST-RAE harness and the same model:
  morgan     : 2048-bit ECFP4 fingerprint            (the fingerprint+GBM baseline)
  generic    : generic RDKit thermo/aromatic descriptors only
  physics2d  : generic + heme/charge motifs + PROTONATION block
  physics3d  : physics2d + 3D geometry (PBF/NPR planarity + basicN–aromatic dist)

Run (fep env):
  PY=python
  $PY scripts/run_ablation.py            # morgan + generic + physics2d (fast)
  $PY scripts/run_ablation.py --with-3d  # adds physics3d (uses feature cache)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rdkit import Chem  # noqa: E402
from rdkit.Chem import rdFingerprintGenerator  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402

from cyp_challenge import config, data, metrics  # noqa: E402
from cyp_challenge.features import (  # noqa: E402
    _GEOM3D_KEYS, _aromatic_planarity_block, _motif_block, _protonation_block,
    _thermo_block, featurize_smiles,
)
from cyp_challenge.splits import scaffold_folds  # noqa: E402

ISOFORMS = config.ISOFORMS
N_SPLITS = 5
CACHE = ROOT / "data" / "processed" / "feat_cache.pkl"

_s = Chem.MolFromSmiles("c1ccccc1CCN")
THERMO = list(_thermo_block(_s)); AROM = list(_aromatic_planarity_block(_s))
MOTIF = list(_motif_block(_s));   PROTON = list(_protonation_block(_s))
GENERIC = THERMO + AROM
PHYSICS2D = GENERIC + MOTIF + PROTON
PHYSICS3D = PHYSICS2D + _GEOM3D_KEYS


def make_model():
    return HistGradientBoostingRegressor(
        loss="absolute_error", learning_rate=0.06, max_iter=400,
        max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.1, random_state=0,
    )


def morgan_matrix(smiles) -> tuple[np.ndarray, np.ndarray]:
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    rows, valid = [], []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if mol is None:
            rows.append(np.zeros(2048)); valid.append(False)
        else:
            rows.append(np.asarray(gen.GetFingerprintAsNumPy(mol))); valid.append(True)
    return np.vstack(rows), np.array(valid)


def cv_st_rae(X: pd.DataFrame, inh: pd.DataFrame, valid: np.ndarray,
              folds: np.ndarray) -> dict[str, float]:
    """Per-isoform scaffold-CV ST-RAE for one feature matrix."""
    res = {}
    for iso in ISOFORMS:
        ycol = f"{iso}_pIC50_direct_inhibition"
        lo, hi = f"{ycol}_conf_low", f"{ycol}_conf_high"
        has = inh[ycol].notna().to_numpy() & valid
        oof = np.full(len(inh), np.nan)
        for f in range(N_SPLITS):
            tr = has & (folds != f); va = has & (folds == f)
            if tr.sum() == 0 or va.sum() == 0:
                continue
            m = make_model(); m.fit(X[tr], inh.loc[tr, ycol])
            oof[va] = m.predict(X[va])
        res[iso] = metrics.st_rae(inh.loc[has, ycol], oof[has],
                                  inh.loc[has, lo], inh.loc[has, hi])
    res["macro"] = float(np.mean([res[i] for i in ISOFORMS]))
    return res


def main() -> None:
    with_3d = "--with-3d" in sys.argv
    inh = data.load_train_inhibition()
    folds = scaffold_folds(inh["SMILES"], n_splits=N_SPLITS, seed=0)

    print("Building feature matrices ...")
    feats = featurize_smiles(inh["SMILES"], include_3d=with_3d,
                             cache_path=str(CACHE) if with_3d else None)
    valid = feats.attrs["valid"]
    Xmorgan, vmorgan = morgan_matrix(inh["SMILES"])

    sets = {
        "morgan":    (Xmorgan, vmorgan),
        "generic":   (feats[GENERIC].to_numpy(), valid),
        "physics2d": (feats[PHYSICS2D].to_numpy(), valid),
    }
    if with_3d:
        sets["physics3d"] = (feats[PHYSICS3D].to_numpy(), valid)
        sets["physics3d+morgan"] = (
            np.hstack([feats[PHYSICS3D].to_numpy(), Xmorgan]), valid & vmorgan)

    results = {}
    for name, (X, v) in sets.items():
        print(f"  scaffold-CV: {name} ({X.shape[1]} features) ...")
        results[name] = cv_st_rae(X, inh, v, folds)

    hdr = f"\n{'feature set':<12} " + " ".join(f"{i[3:]:>7}" for i in ISOFORMS) + "   macro"
    print("\n=== Scaffold-CV ST-RAE (lower=better) ==="); print(hdr)
    for name, r in results.items():
        print(f"{name:<12} " + " ".join(f"{r[i]:7.3f}" for i in ISOFORMS)
              + f"  {r['macro']:6.3f}")

    print("\n=== Diagnostic deltas (negative = improvement) ===")
    print(f"  CYP2D6 protonation effect (physics2d − generic): "
          f"{results['physics2d']['CYP2D6'] - results['generic']['CYP2D6']:+.3f}")
    print(f"  physics2d vs morgan (macro):                     "
          f"{results['physics2d']['macro'] - results['morgan']['macro']:+.3f}")
    if with_3d:
        print(f"  CYP1A2 3D-planarity effect (physics3d − physics2d): "
              f"{results['physics3d']['CYP1A2'] - results['physics2d']['CYP1A2']:+.3f}")


if __name__ == "__main__":
    main()

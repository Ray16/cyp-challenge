"""Diagnose and fix range compression.

Q1: does scaffold-CV reproduce the compression (OOF pred SD << train label SD)?
    If yes, CV can guide the fix. If OOF is fine but the TEST submission is
    compressed, it's pure OOD collapse and we need a harsher, shift-aware split.
Q2: is the blinded test chemically OOD vs train (nearest-neighbour Tanimoto)?
Fixes tried under the correct (vendored) metric:
    - variance matching: stretch predictions to the train label SD (rank-preserving)
    - Ridge (a linear model EXTRAPOLATES, unlike trees that revert to the mean)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rdkit import Chem  # noqa: E402
from rdkit.Chem import rdFingerprintGenerator  # noqa: E402
from rdkit.DataStructs import BulkTanimotoSimilarity  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from cyp_challenge import config, data, metrics  # noqa: E402
from cyp_challenge.features import featurize_smiles  # noqa: E402
from cyp_challenge.splits import scaffold_folds  # noqa: E402

ISO = config.ISOFORMS
CACHE = str(ROOT / "data" / "processed" / "feat_cache.pkl")


def gbm():
    return HistGradientBoostingRegressor(
        loss="absolute_error", learning_rate=0.06, max_iter=400, max_leaf_nodes=31,
        min_samples_leaf=20, l2_regularization=1.0, early_stopping=True, random_state=0)


def ridge():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0))


def oof(estimator_fn, X, y, has, folds):
    p = np.full(len(y), np.nan)
    for f in range(5):
        tr = has & (folds != f); va = has & (folds == f)
        m = estimator_fn(); m.fit(X[tr], y[tr]); p[va] = m.predict(X[va])
    return p


def test_ood_distance():
    inh = data.load_train_inhibition(); test = data.load_test_blinded()
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    def fps(smis):
        return [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in smis
                if Chem.MolFromSmiles(s) is not None]
    tr_fps = fps(inh["SMILES"]); te_fps = fps(test["SMILES"])
    nn = [max(BulkTanimotoSimilarity(f, tr_fps)) for f in te_fps]
    nn = np.array(nn)
    print(f"Test→train nearest-neighbour Tanimoto: median {np.median(nn):.2f}, "
          f"frac<0.4 (dissimilar) {np.mean(nn < 0.4):.2f}, frac<0.3 {np.mean(nn < 0.3):.2f}")


def main():
    inh = data.load_train_inhibition()
    X = featurize_smiles(inh["SMILES"], include_3d=True, cache_path=CACHE)
    Xn = X.to_numpy(); valid = X.attrs["valid"]
    folds = scaffold_folds(inh["SMILES"], 5, 0)

    print("=== Q2: is the test OOD? ===")
    test_ood_distance()

    print("\n=== Q1: compression in scaffold-CV OOF, and does decompression help? ===")
    print(f"{'iso':>6} {'trainSD':>7} {'gbmSD':>6} {'ridgeSD':>7} | "
          f"{'gbm':>6} {'gbm+vm':>6} {'ridge':>6}  (ST-RAE)")
    macro = {"gbm": [], "vm": [], "ridge": []}
    for iso in ISO:
        yc = f"{iso}_pIC50_direct_inhibition"
        y = inh[yc].to_numpy()
        lo = inh[f"{yc}_conf_low"].to_numpy(); hi = inh[f"{yc}_conf_high"].to_numpy()
        has = inh[yc].notna().to_numpy() & valid
        yv = y[has]; lov, hiv = lo[has], hi[has]; m = yv.mean(); tsd = yv.std()

        og = oof(gbm, Xn, y, has, folds)[has]
        orr = oof(ridge, Xn, y, has, folds)[has]
        # variance-matched GBM (rank-preserving stretch to train SD)
        ovm = m + (og - m) * (tsd / og.std())

        s_g = metrics.st_rae(yv, og, lov, hiv)
        s_vm = metrics.st_rae(yv, ovm, lov, hiv)
        s_r = metrics.st_rae(yv, orr, lov, hiv)
        macro["gbm"].append(s_g); macro["vm"].append(s_vm); macro["ridge"].append(s_r)
        print(f"{iso[3:]:>6} {tsd:7.2f} {og.std():6.2f} {orr.std():7.2f} | "
              f"{s_g:6.3f} {s_vm:6.3f} {s_r:6.3f}")
    print(f"{'macro':>6} {'':7} {'':6} {'':7} | "
          f"{np.mean(macro['gbm']):6.3f} {np.mean(macro['vm']):6.3f} {np.mean(macro['ridge']):6.3f}")


if __name__ == "__main__":
    main()

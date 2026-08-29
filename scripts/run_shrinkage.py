"""Does shrinking predictions toward the mean help under the CORRECT ST-RAE?

The official metric normalizes by the soft-thresholded mean-baseline, so it
punishes prediction variance on near-mean/wide-CI compounds. This sweeps a
shrinkage factor alpha: pred_shrunk = mean + alpha*(pred - mean). alpha<1 pulls
toward the mean. If the optimum is <1, we're over-confident and eating that penalty.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402

from cyp_challenge import config, data, metrics  # noqa: E402
from cyp_challenge.features import featurize_smiles  # noqa: E402
from cyp_challenge.splits import scaffold_folds  # noqa: E402

ISO = config.ISOFORMS
CACHE = str(ROOT / "data" / "processed" / "feat_cache.pkl")


def model():
    return HistGradientBoostingRegressor(
        loss="absolute_error", learning_rate=0.06, max_iter=400, max_leaf_nodes=31,
        min_samples_leaf=20, l2_regularization=1.0, early_stopping=True, random_state=0)


def main():
    inh = data.load_train_inhibition()
    X = featurize_smiles(inh["SMILES"], include_3d=True, cache_path=CACHE)
    valid = X.attrs["valid"]; cols = list(X.columns)
    folds = scaffold_folds(inh["SMILES"], 5, 0)

    alphas = np.linspace(0.0, 1.3, 27)
    per_iso_best = {}
    macro_by_alpha = np.zeros_like(alphas)
    for iso in ISO:
        yc = f"{iso}_pIC50_direct_inhibition"
        lo, hi = inh[f"{yc}_conf_low"], inh[f"{yc}_conf_high"]
        has = inh[yc].notna().to_numpy() & valid
        oof = np.full(len(inh), np.nan)
        for f in range(5):
            tr = has & (folds != f); va = has & (folds == f)
            m = model(); m.fit(X.loc[tr, cols], inh.loc[tr, yc])
            oof[va] = m.predict(X.loc[va, cols])
        mean = inh.loc[has, yc].mean()
        y, ll, hh = inh.loc[has, yc].to_numpy(), lo[has].to_numpy(), hi[has].to_numpy()
        scores = [metrics.st_rae(y, mean + a * (oof[has] - mean), ll, hh) for a in alphas]
        bi = int(np.argmin(scores))
        per_iso_best[iso] = (alphas[bi], scores[bi], scores[-1] if alphas[-1] == 1.0 else None)
        # ST-RAE at alpha=1.0 (no shrink) for reference
        s1 = metrics.st_rae(y, oof[has], ll, hh)
        macro_by_alpha += np.array(scores) / len(ISO)
        print(f"  {iso}: no-shrink(α=1)={s1:.3f}  best α={alphas[bi]:.2f} → {scores[bi]:.3f}")

    bi = int(np.argmin(macro_by_alpha))
    a1 = int(np.argmin(np.abs(alphas - 1.0)))
    print(f"\n  MACRO: no-shrink={macro_by_alpha[a1]:.3f}  "
          f"best α={alphas[bi]:.2f} → {macro_by_alpha[bi]:.3f}")


if __name__ == "__main__":
    main()

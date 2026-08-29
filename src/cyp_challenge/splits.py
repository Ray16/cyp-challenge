"""Scaffold-aware cross-validation.

The public leaderboard scores only ~50% of the test set; the blinded half may hold
distinct chemotypes. A random split flatters us by leaking scaffolds between folds.
We therefore build folds that are DISJOINT in Bemis-Murcko scaffold, so internal CV
estimates generalization to novel chemistry — the quantity the challenge rewards.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import murcko_scaffold


def scaffold_folds(smiles: pd.Series, n_splits: int = 5,
                   seed: int = 0) -> np.ndarray:
    """Assign each row a fold id (0..n_splits-1) such that all molecules sharing a
    Murcko scaffold land in the same fold. Scaffolds are distributed largest-first
    across the currently-smallest folds to keep fold sizes balanced.

    Returns an int array of fold ids aligned to `smiles`.
    """
    scaffolds = smiles.map(murcko_scaffold)
    # empty-scaffold (parse failures / acyclic) get a unique key so they don't all
    # clump into one giant pseudo-scaffold group.
    groups: dict[str, list[int]] = {}
    for i, sc in enumerate(scaffolds):
        key = sc if sc else f"__singleton_{i}"
        groups.setdefault(key, []).append(i)

    fold_id = np.full(len(smiles), -1, dtype=int)
    fold_sizes = np.zeros(n_splits, dtype=int)
    rng = np.random.default_rng(seed)
    ordered = sorted(groups.values(), key=lambda idx: (-len(idx), rng.random()))
    for idx in ordered:
        f = int(np.argmin(fold_sizes))
        fold_id[idx] = f
        fold_sizes[f] += len(idx)
    assert (fold_id >= 0).all()
    return fold_id

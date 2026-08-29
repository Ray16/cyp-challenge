"""Multi-task neural net to lift RANK (the ceiling after recalibration).

Shared trunk + per-task heads, trained on all four pIC50 endpoints jointly with a
masked loss (sparse label matrix → each compound contributes only its measured
tasks). Cross-task borrowing is the point: CYP2D6 (our weakest, OOF ρ=0.38) can
learn from the correlated isoforms. The DENSE single-concentration log2fc (4 more
heads) is added as auxiliary tasks to regularize the shared representation —
mirroring the frontier's chemprop-mt-aux.

We optimize/measure RANK (OOF Pearson), not ST-RAE: recalibration already handles
placement, and rank is the pitfall-free objective.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from rdkit import Chem  # noqa: E402
from rdkit.Chem import rdFingerprintGenerator  # noqa: E402
from scipy.stats import pearsonr  # noqa: E402

from cyp_challenge import config, data  # noqa: E402
from cyp_challenge.features import featurize_smiles  # noqa: E402
from cyp_challenge.splits import scaffold_folds  # noqa: E402

ISO = config.ISOFORMS
CACHE = str(ROOT / "data" / "processed" / "feat_cache.pkl")
GBM_RHO = {"CYP1A2": 0.48, "CYP2C9": 0.60, "CYP2D6": 0.38, "CYP3A4": 0.74}
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)


class MTL(nn.Module):
    def __init__(self, d_in, n_tasks, hidden=(512, 256), p=0.3):
        super().__init__()
        layers, d = [], d_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(p)]
            d = h
        self.trunk = nn.Sequential(*layers)
        self.heads = nn.Linear(d, n_tasks)

    def forward(self, x):
        return self.heads(self.trunk(x))


def morgan(smis):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    return np.vstack([np.asarray(gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s))) for s in smis])


def train_predict(Xtr, Ytr, Mtr, Xte, n_tasks, epochs=120):
    """Train masked-loss MTL on (Xtr,Ytr,mask Mtr), return predictions on Xte."""
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=DEV)
    Ytr_t = torch.tensor(np.nan_to_num(Ytr), dtype=torch.float32, device=DEV)
    Mtr_t = torch.tensor(Mtr, dtype=torch.float32, device=DEV)
    Xte_t = torch.tensor(Xte, dtype=torch.float32, device=DEV)
    net = MTL(Xtr.shape[1], n_tasks).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    bs = 256
    for ep in range(epochs):
        net.train(); perm = torch.randperm(len(Xtr_t), device=DEV)
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            pred = net(Xtr_t[idx])
            m = Mtr_t[idx]
            loss = (((pred - Ytr_t[idx]) ** 2) * m).sum() / m.sum().clamp(min=1)
            loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        return net(Xte_t).cpu().numpy()


def main():
    inh = data.load_train_inhibition()
    sc = data.load_train_single_concentration()
    # pivot single-conc log2fc to wide (aux tasks), align to inh by Molecule_Name
    scw = sc.pivot_table(index="Molecule_Name", columns="enzyme",
                         values="log2fc_estimate", aggfunc="mean")
    scw = scw.reindex(inh["Molecule_Name"]).reset_index(drop=True)

    feats = featurize_smiles(inh["SMILES"], include_3d=True, cache_path=CACHE)
    dense = feats.to_numpy()
    dense = np.nan_to_num(dense, nan=np.nanmedian(dense))
    X = np.hstack([dense, morgan(inh["SMILES"])]).astype(np.float32)

    # target matrix: 4 pIC50 (primary) + 4 log2fc (aux)
    pic = np.column_stack([inh[f"{i}_pIC50_direct_inhibition"].to_numpy() for i in ISO])
    aux = np.column_stack([scw[i].to_numpy() if i in scw.columns else np.full(len(inh), np.nan)
                           for i in ISO])
    Y = np.hstack([pic, aux])
    # standardize each task column (for balanced loss); rank is scale-invariant
    mu = np.nanmean(Y, 0); sd = np.nanstd(Y, 0); sd[sd == 0] = 1
    Yz = (Y - mu) / sd
    M = (~np.isnan(Y)).astype(np.float32)

    # standardize dense feature block
    dmu = X[:, :dense.shape[1]].mean(0); dstd = X[:, :dense.shape[1]].std(0); dstd[dstd == 0] = 1

    folds = scaffold_folds(inh["SMILES"], 5, 0)
    oof = np.full((len(inh), 8), np.nan)
    for f in range(5):
        tr = folds != f; te = folds == f
        Xtr, Xte = X.copy(), X.copy()
        Xtr[:, :dense.shape[1]] = (X[:, :dense.shape[1]] - dmu) / dstd
        Xte[:, :dense.shape[1]] = (X[:, :dense.shape[1]] - dmu) / dstd
        pred = train_predict(Xtr[tr], Yz[tr], M[tr], Xte[te], n_tasks=8)
        oof[te] = pred
    print(f"device={DEV}\n{'iso':>7} {'GBM ρ':>6} {'MTL ρ':>6} {'Δ':>6}")
    deltas = []
    for j, iso in enumerate(ISO):
        obs = ~np.isnan(pic[:, j])
        rho = pearsonr(pic[obs, j], oof[obs, j])[0]
        deltas.append(rho - GBM_RHO[iso])
        print(f"{iso[3:]:>7} {GBM_RHO[iso]:6.2f} {rho:6.2f} {rho-GBM_RHO[iso]:+6.2f}")
    print(f"{'mean':>7} {np.mean(list(GBM_RHO.values())):6.2f} "
          f"{np.mean([pearsonr(pic[~np.isnan(pic[:,j]),j], oof[~np.isnan(pic[:,j]),j])[0] for j in range(4)]):6.2f} "
          f"{np.mean(deltas):+6.2f}")


if __name__ == "__main__":
    main()

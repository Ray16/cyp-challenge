"""Self-contained D-MPNN (ChemProp-style) multi-task model with physics aux features.

Directed message passing over molecular graphs LEARNS a representation, which our
earlier experiment showed is the true rank ceiling (fixed descriptors capped CYP2D6
at ρ≈0.39; the frontier's ChemProp reaches 0.68). Our physics descriptors ride along
as auxiliary molecule-level features (the frontier's chemprop-mt-aux recipe), and the
dense single-conc log2fc are extra task heads. Masked multi-task loss over the sparse
label matrix. We report OOF Pearson (rank) — placement is handled by recalibration.

No torch_geometric dependency: custom index-based batching. Runs on GPU if present.
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
from scipy.stats import pearsonr  # noqa: E402

from cyp_challenge import config, data  # noqa: E402
from cyp_challenge.features import featurize_smiles  # noqa: E402
from cyp_challenge.splits import scaffold_folds  # noqa: E402

ISO = config.ISOFORMS
CACHE = str(ROOT / "data" / "processed" / "feat_cache.pkl")
GBM_RHO = {"CYP1A2": 0.48, "CYP2C9": 0.60, "CYP2D6": 0.38, "CYP3A4": 0.74}
MLP_RHO = {"CYP1A2": 0.50, "CYP2C9": 0.64, "CYP2D6": 0.39, "CYP3A4": 0.77}
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); np.random.seed(0)

ELEMENTS = [5, 6, 7, 8, 9, 15, 16, 17, 35, 53]  # B C N O F P S Cl Br I


def _onehot(v, choices):
    return [float(v == c) for c in choices] + [float(v not in choices)]


def atom_feats(a):
    return (_onehot(a.GetAtomicNum(), ELEMENTS)
            + _onehot(a.GetTotalDegree(), [0, 1, 2, 3, 4, 5])
            + _onehot(a.GetFormalCharge(), [-1, 0, 1])
            + _onehot(a.GetTotalNumHs(), [0, 1, 2, 3, 4])
            + _onehot(int(a.GetHybridization()), [2, 3, 4, 5, 6])
            + [float(a.GetIsAromatic()), float(a.IsInRing()), a.GetMass() * 0.01])


def bond_feats(b):
    bt = b.GetBondType()
    return [float(bt == Chem.BondType.SINGLE), float(bt == Chem.BondType.DOUBLE),
            float(bt == Chem.BondType.TRIPLE), float(bt == Chem.BondType.AROMATIC),
            float(b.GetIsConjugated()), float(b.IsInRing())]


D_ATOM = len(atom_feats(Chem.MolFromSmiles("CC").GetAtomWithIdx(0)))
D_BOND = 6


def mol_graph(smi):
    m = Chem.MolFromSmiles(smi)
    fa = np.array([atom_feats(a) for a in m.GetAtoms()], dtype=np.float32)
    src, tgt, fb, rev = [], [], [], []
    for b in m.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bf = bond_feats(b)
        e = len(src)
        src += [i, j]; tgt += [j, i]; fb += [bf, bf]; rev += [e + 1, e]
    if not src:  # single-atom / no bonds
        return fa, (np.zeros((0,), int), np.zeros((0,), int),
                    np.zeros((0, D_BOND), np.float32), np.zeros((0,), int))
    return fa, (np.array(src), np.array(tgt), np.array(fb, np.float32), np.array(rev))


class DMPNN(nn.Module):
    def __init__(self, d_aux, n_tasks, hidden=300, depth=4, p=0.2):
        super().__init__()
        self.depth = hidden and depth
        self.W_i = nn.Linear(D_ATOM + D_BOND, hidden, bias=False)
        self.W_h = nn.Linear(hidden, hidden, bias=False)
        self.W_o = nn.Linear(D_ATOM + hidden, hidden)
        self.drop = nn.Dropout(p)
        self.ffn = nn.Sequential(
            nn.Linear(hidden + d_aux, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Dropout(p), nn.Linear(hidden, n_tasks))

    def forward(self, fa, src, tgt, fb, rev, atom2mol, n_mol, aux):
        h0 = torch.relu(self.W_i(torch.cat([fa[src], fb], 1)))  # [E, H]
        h = h0
        for _ in range(self.depth - 1):
            agg = torch.zeros(fa.size(0), h.size(1), device=fa.device).index_add_(0, tgt, h)
            msg = agg[src] - h[rev]
            h = self.drop(torch.relu(h0 + self.W_h(msg)))
        agg = torch.zeros(fa.size(0), h.size(1), device=fa.device).index_add_(0, tgt, h)
        atom_h = torch.relu(self.W_o(torch.cat([fa, agg], 1)))           # [A, H]
        mol = torch.zeros(n_mol, atom_h.size(1), device=fa.device).index_add_(0, atom2mol, atom_h)
        counts = torch.zeros(n_mol, 1, device=fa.device).index_add_(
            0, atom2mol, torch.ones(fa.size(0), 1, device=fa.device)).clamp(min=1)
        mol = mol / counts
        return self.ffn(torch.cat([mol, aux], 1))


def collate(graphs, aux, idx):
    fa, src, tgt, fb, rev, a2m = [], [], [], [], [], []
    atom_off, edge_off = 0, 0
    for k, gi in enumerate(idx):
        f, (s, t, b, r) = graphs[gi]
        fa.append(f); a2m.append(np.full(len(f), k))
        if len(s):
            src.append(s + atom_off); tgt.append(t + atom_off)
            fb.append(b); rev.append(r + edge_off)
        atom_off += len(f); edge_off += len(s)
    return _pack(fa, src, tgt, fb, rev, a2m, aux[idx], len(idx))


def _pack(fa, src, tgt, fb, rev, a2m, aux, n_mol):
    fa_t = torch.tensor(np.concatenate(fa), dtype=torch.float32, device=DEV)
    a2m_t = torch.tensor(np.concatenate(a2m), dtype=torch.long, device=DEV)
    if src:
        src_t = torch.tensor(np.concatenate(src), dtype=torch.long, device=DEV)
        tgt_t = torch.tensor(np.concatenate(tgt), dtype=torch.long, device=DEV)
        fb_t = torch.tensor(np.concatenate(fb), dtype=torch.float32, device=DEV)
        rev_t = torch.tensor(np.concatenate(rev), dtype=torch.long, device=DEV)
    else:
        src_t = tgt_t = rev_t = torch.zeros(0, dtype=torch.long, device=DEV)
        fb_t = torch.zeros((0, D_BOND), dtype=torch.float32, device=DEV)
    aux_t = torch.tensor(aux, dtype=torch.float32, device=DEV)
    return fa_t, src_t, tgt_t, fb_t, rev_t, a2m_t, n_mol, aux_t


def main():
    inh = data.load_train_inhibition()
    sc = data.load_train_single_concentration()
    scw = sc.pivot_table(index="Molecule_Name", columns="enzyme",
                         values="log2fc_estimate", aggfunc="mean").reindex(
                         inh["Molecule_Name"]).reset_index(drop=True)

    print("Building molecular graphs ...")
    graphs = [mol_graph(s) for s in inh["SMILES"]]

    aux = featurize_smiles(inh["SMILES"], include_3d=True, cache_path=CACHE).to_numpy()
    aux = np.nan_to_num(aux, nan=np.nanmedian(aux)).astype(np.float32)

    pic = np.column_stack([inh[f"{i}_pIC50_direct_inhibition"].to_numpy() for i in ISO])
    auxlab = np.column_stack([scw[i].to_numpy() if i in scw.columns else np.full(len(inh), np.nan)
                              for i in ISO])
    Y = np.hstack([pic, auxlab]); mu = np.nanmean(Y, 0); sd = np.nanstd(Y, 0); sd[sd == 0] = 1
    Yz = torch.tensor(np.nan_to_num((Y - mu) / sd), dtype=torch.float32, device=DEV)
    Mk = torch.tensor((~np.isnan(Y)).astype(np.float32), device=DEV)

    folds = scaffold_folds(inh["SMILES"], 5, 0)
    oof = np.full((len(inh), 8), np.nan)
    for f in range(5):
        tr_idx = np.where(folds != f)[0]; te_idx = np.where(folds == f)[0]
        amu = aux[tr_idx].mean(0); asd = aux[tr_idx].std(0); asd[asd == 0] = 1
        auxn = (aux - amu) / asd
        net = DMPNN(d_aux=aux.shape[1], n_tasks=8).to(DEV)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
        for ep in range(80):
            net.train(); np.random.shuffle(tr_idx)
            for i in range(0, len(tr_idx), 50):
                b = tr_idx[i:i + 50]
                batch = collate(graphs, auxn, b)
                opt.zero_grad()
                pred = net(*batch)
                loss = (((pred - Yz[b]) ** 2) * Mk[b]).sum() / Mk[b].sum().clamp(min=1)
                loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            for i in range(0, len(te_idx), 256):
                b = te_idx[i:i + 256]
                oof[b] = net(*collate(graphs, auxn, b)).cpu().numpy()
        print(f"  fold {f} done")

    print(f"\ndevice={DEV}\n{'iso':>7} {'GBM':>5} {'MLP':>5} {'DMPNN':>6} {'Δvsbest':>7}")
    for j, iso in enumerate(ISO):
        obs = ~np.isnan(pic[:, j])
        rho = pearsonr(pic[obs, j], oof[obs, j])[0]
        best = max(GBM_RHO[iso], MLP_RHO[iso])
        print(f"{iso[3:]:>7} {GBM_RHO[iso]:5.2f} {MLP_RHO[iso]:5.2f} {rho:6.2f} {rho-best:+7.2f}")
    macro = np.mean([pearsonr(pic[~np.isnan(pic[:, j]), j], oof[~np.isnan(pic[:, j]), j])[0]
                     for j in range(4)])
    print(f"{'mean':>7} {0.55:5.2f} {0.57:5.2f} {macro:6.2f}")


if __name__ == "__main__":
    main()

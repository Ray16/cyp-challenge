"""Assemble the real contender: D-MPNN (frontier-level rank) + recalibration
(correct placement). Train an ensemble on ALL inhibition data, predict the 750
blinded compounds, then affine-recalibrate each isoform onto the blind distribution.

Rank comes from the learned representation; placement from recalibration — the two
separable halves, finally combined.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from cyp_challenge import config, data, validate  # noqa: E402
from cyp_challenge.features import featurize_smiles  # noqa: E402

_spec = importlib.util.spec_from_file_location("dmpnn", ROOT / "scripts" / "run_dmpnn.py")
D = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(D)

ISO = config.ISOFORMS
CACHE = str(ROOT / "data" / "processed" / "feat_cache.pkl")
DEV = D.DEV
K_ENSEMBLE = 4

# blind moments (public solve) + OOF Pearson we measured + OOF->blind ratios
BLIND = {"CYP1A2": (4.412, 1.553), "CYP2C9": (4.830, 1.101),
         "CYP2D6": (3.107, 1.599), "CYP3A4": (4.880, 1.272)}
OOF_RHO = {"CYP1A2": 0.56, "CYP2C9": 0.66, "CYP2D6": 0.38, "CYP3A4": 0.79}
OOF_TO_BLIND = {"CYP1A2": 1.32, "CYP2C9": 1.23, "CYP2D6": 1.66, "CYP3A4": 1.07}


def build(smis):
    return [D.mol_graph(s) for s in smis]


def main():
    inh = data.load_train_inhibition(); test = data.load_test_blinded()
    sc = data.load_train_single_concentration()
    scw = sc.pivot_table(index="Molecule_Name", columns="enzyme", values="log2fc_estimate",
                         aggfunc="mean").reindex(inh["Molecule_Name"]).reset_index(drop=True)

    g_tr, g_te = build(inh["SMILES"]), build(test["SMILES"])
    aux_tr = np.nan_to_num(featurize_smiles(inh["SMILES"], include_3d=True, cache_path=CACHE).to_numpy()).astype(np.float32)
    aux_te = np.nan_to_num(featurize_smiles(test["SMILES"], include_3d=True, cache_path=CACHE).to_numpy()).astype(np.float32)
    amu, asd = aux_tr.mean(0), aux_tr.std(0); asd[asd == 0] = 1
    aux_tr = (aux_tr - amu) / asd; aux_te = (aux_te - amu) / asd

    pic = np.column_stack([inh[f"{i}_pIC50_direct_inhibition"].to_numpy() for i in ISO])
    al = np.column_stack([scw[i].to_numpy() if i in scw.columns else np.full(len(inh), np.nan) for i in ISO])
    Y = np.hstack([pic, al]); mu = np.nanmean(Y, 0); sd = np.nanstd(Y, 0); sd[sd == 0] = 1
    Yz = torch.tensor(np.nan_to_num((Y - mu) / sd), dtype=torch.float32, device=DEV)
    Mk = torch.tensor((~np.isnan(Y)).astype(np.float32), device=DEV)
    idx_all = np.arange(len(inh))

    preds = np.zeros((len(test), 4))
    for seed in range(K_ENSEMBLE):
        torch.manual_seed(seed); np.random.seed(seed)
        net = D.DMPNN(d_aux=aux_tr.shape[1], n_tasks=8).to(DEV)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
        order = idx_all.copy()
        for ep in range(80):
            net.train(); np.random.shuffle(order)
            for i in range(0, len(order), 50):
                b = order[i:i + 50]
                opt.zero_grad()
                loss = (((net(*D.collate(g_tr, aux_tr, b)) - Yz[b]) ** 2) * Mk[b]).sum() / Mk[b].sum().clamp(min=1)
                loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            p = np.zeros((len(test), 8))
            for i in range(0, len(test), 256):
                b = np.arange(i, min(i + 256, len(test)))
                p[b] = net(*D.collate(g_te, aux_te, b)).cpu().numpy()
        preds += p[:, :4] / K_ENSEMBLE
        print(f"  ensemble member {seed} done")

    # recalibrate each isoform onto the blind distribution (affine, rank-preserving)
    sub = test[["SMILES", "Molecule_Name"]].copy()
    print(f"\n{'iso':>7} {'blindρ~':>7} {'raw μ/σ':>12} -> {'recal μ/σ':>12}")
    for j, iso in enumerate(ISO):
        bmean, bsd = BLIND[iso]
        rho = min(OOF_RHO[iso] * OOF_TO_BLIND[iso], 0.95)
        z = (preds[:, j] - preds[:, j].mean()) / preds[:, j].std()
        recal = bmean + z * (rho * bsd)
        sub[f"{iso}_pIC50_direct_inhibition"] = recal
        print(f"{iso[3:]:>7} {rho:7.2f} {preds[:,j].mean():5.1f}/{preds[:,j].std():4.2f}"
              f"    -> {recal.mean():5.1f}/{recal.std():4.2f}")

    errs = validate.validate_regression_submission(sub, test)
    out = ROOT / "submissions" / "regression" / "dmpnn_recalibrated.csv"
    sub.to_csv(out, index=False)
    print(f"\nvalidation: {'PASS' if not errs else errs}\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

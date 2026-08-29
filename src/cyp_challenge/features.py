"""Physics-informed molecular featurization for CYP inhibition.

The premise (see the approach memo): pIC50 is, to first order, a protein-ligand
binding free energy, ΔG_bind = -2.303 RT * pKi ≈ -2.303 RT * (pIC50 - const). So
instead of an opaque 2048-bit fingerprint we build descriptors that map onto the
physical terms that drive ΔG_bind, plus mechanistic motifs specific to each CYP
pocket and to the P450 catalytic cycle.

Feature blocks
--------------
1. thermodynamic drivers  - size, lipophilicity, polarity, H-bonding, flexibility,
   polarizability (MolMR) — the generic ΔG_bind decomposition.
2. protonation / charge   - basic and acidic centers at physiological pH; drives
   CYP2D6 (Asp301 salt bridge with a basic amine) and CYP2C9 (Arg108 with acids).
3. heme coordination      - accessible sp2 N lone pairs (imidazole/triazole/pyridine)
   and nitriles → Type II inhibition, the strongest single potency motif (azoles).
4. aromatic / planarity   - flat polyaromatic character → CYP1A2's narrow planar slot.
5. bioactivation alerts   - SMARTS toxicophores that generate reactive metabolites →
   mechanism-based (time-dependent) inhibition, for the TDI classification track.

Everything here is 2D/topological and fast; a true 3D planarity (PMI) and QM-derived
descriptors are deferred to a later tier.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import (
    AllChem,
    Crippen,
    Descriptors,
    Descriptors3D,
    Lipinski,
    rdMolDescriptors,
)
from rdkit.Chem.Scaffolds import MurckoScaffold

PH = 7.4  # assay / physiological pH for protonation-state modeling

# --- SMARTS motif definitions -------------------------------------------------

# Accessible aromatic sp2 nitrogen with a lone pair available to ligate the heme
# iron (Type II binding). Pyridine-like N, imidazole/triazole "pyridine-type" N,
# and nitrile. Excludes pyrrole-type N (lone pair in the ring, not available).
HEME_COORD_SMARTS = {
    "pyridine_n": "[nX2;!$(n[#1])]",          # aromatic N, 2 connections (pyridine-type)
    "imidazole": "c1cnc[nH0,nH1]1",
    "triazole": "c1n[nH0,nH1]nc1",
    "nitrile": "[NX1]#[CX2]",
    "aromatic_amine_free": "[$([NX3;H2,H1;!$(NC=O)])][c]",  # aniline-type (weaker)
}

# Basic centers protonated at pH 7.4 (net positive) — CYP2D6 salt-bridge partner.
BASIC_SMARTS = {
    "aliphatic_amine": "[NX3;H2,H1,H0;!$(NC=O);!$(N=*);!$([N+]);!$(Nc)]",
    "amidine": "[NX3][CX3]=[NX2]",
    "guanidine": "[NX3][CX3](=[NX2])[NX3]",
}

# Acidic centers deprotonated at pH 7.4 (net negative) — CYP2C9 / Arg108 partner.
ACIDIC_SMARTS = {
    "carboxylic_acid": "[CX3](=O)[OX2H1]",
    "tetrazole": "c1n[nH]nn1",
    "acyl_sulfonamide": "[SX4](=O)(=O)[NX3][CX3]=O",
    "sulfonic_acid": "[SX4](=O)(=O)[OX2H1]",
}

# Bioactivation / reactive-metabolite alerts → mechanism-based (time-dependent)
# inhibition. Classic P450 MBI liabilities.
TDI_ALERT_SMARTS = {
    "furan": "c1ccoc1",
    "thiophene": "c1ccsc1",
    "terminal_alkyne": "[CX2]#[CX1H]",
    "terminal_alkene": "[CX3H2]=[CX3H1]",
    "methylenedioxyphenyl": "c1ccc2c(c1)OCO2",
    "aniline": "[NX3;H2,H1][c]",
    "nitroaromatic": "[c][NX3](=O)=O",
    "hydrazine": "[NX3][NX3]",
    "thiourea": "[NX3][CX3](=[SX1])[NX3]",
    "tertiary_aliphatic_amine": "[NX3;H0;!$(NC=O);!$(N=*);!$(Nc)]",  # → MI complex
    "epoxide": "[OX2r3]1[#6r3][#6r3]1",
}

_ALL_SMARTS = {
    **{f"heme_{k}": v for k, v in HEME_COORD_SMARTS.items()},
    **{f"basic_{k}": v for k, v in BASIC_SMARTS.items()},
    **{f"acid_{k}": v for k, v in ACIDIC_SMARTS.items()},
    **{f"tdi_{k}": v for k, v in TDI_ALERT_SMARTS.items()},
}
_COMPILED = {name: Chem.MolFromSmarts(s) for name, s in _ALL_SMARTS.items()}
_bad = [n for n, p in _COMPILED.items() if p is None]
if _bad:
    raise ValueError(f"Invalid SMARTS patterns: {_bad}")

# --- Ionizable groups with representative pKa -------------------------------
# The physically-correct feature for CYP2D6 (Asp301 salt bridge) is not "has a
# nitrogen" but "carries a PROTONATED basic center at pH 7.4"; for CYP2C9 (Arg108)
# it is "carries a deprotonated acid". We estimate the charge state per group via
# Henderson-Hasselbalch with literature pKa values. This is a fast, transparent
# approximation (independent groups, representative pKa) — swap in dimorphite-dl /
# pypka / xtb for microstate rigor later.
BASIC_PKA = {  # protonated form is the cation
    "prim_amine": ("[NX3;H2;!$(NC=O);!$(N=*);!$(Na);!$([N+])]", 10.6),
    "sec_amine":  ("[NX3;H1;!$(NC=O);!$(N=*);!$(Na);!$([N+])]", 10.5),
    "tert_amine": ("[NX3;H0;!$(NC=O);!$(N=*);!$(Na);!$([N+]);!$([n])]", 9.8),
    "amidine":    ("[NX3][CX3]=[NX2;!$([n])]", 12.4),
    "guanidine":  ("[NX3][CX3](=[NX2])[NX3]", 13.6),
    "imidazole":  ("c1cnc[nH0,nH1]1", 6.9),
    "pyridine":   ("[nX2;!$(n[#1])]", 5.2),
}
ACIDIC_PKA = {  # deprotonated form is the anion
    "carboxyl":       ("[CX3](=O)[OX2H1]", 4.2),
    "tetrazole":      ("c1[nH1]nnn1", 4.9),
    "acyl_sulfonamide": ("[SX4](=O)(=O)[NX3H1][CX3]=O", 5.5),
    "sulfonic":       ("[SX4](=O)(=O)[OX2H1]", -1.0),
    "phenol":         ("[c][OX2H1]", 9.9),
}
_BASIC_C = {k: (Chem.MolFromSmarts(s), pka) for k, (s, pka) in BASIC_PKA.items()}
_ACIDIC_C = {k: (Chem.MolFromSmarts(s), pka) for k, (s, pka) in ACIDIC_PKA.items()}


def _frac_protonated_base(pka: float) -> float:
    return 1.0 / (1.0 + 10 ** (PH - pka))


def _frac_deprotonated_acid(pka: float) -> float:
    return 1.0 / (1.0 + 10 ** (pka - PH))


def _protonation_block(mol: Chem.Mol) -> dict[str, float]:
    """Charge state at pH 7.4 — drives CYP2D6 (cationic amine ↔ Asp301) and
    CYP2C9 (anion ↔ Arg108). Also yields logD (charge-corrected logP)."""
    pos, ratio_sum, strongest_base = 0.0, 0.0, 0.0
    n_basic = 0
    for _, (patt, pka) in _BASIC_C.items():
        m = len(mol.GetSubstructMatches(patt))
        if m:
            n_basic += m
            pos += m * _frac_protonated_base(pka)
            ratio_sum += m * 10 ** (pka - PH)
            strongest_base = max(strongest_base, pka)
    neg, n_acidic = 0.0, 0
    for _, (patt, pka) in _ACIDIC_C.items():
        m = len(mol.GetSubstructMatches(patt))
        if m:
            n_acidic += m
            neg += m * _frac_deprotonated_acid(pka)
            ratio_sum += m * 10 ** (PH - pka)
    f_neutral = 1.0 / (1.0 + ratio_sum)
    logp = Crippen.MolLogP(mol)
    return {
        "n_basic_centers": n_basic,
        "n_acidic_centers": n_acidic,
        "pos_charge_pH74": pos,
        "neg_charge_pH74": neg,
        "net_charge_pH74": pos - neg,
        "strongest_base_pKa": strongest_base,
        "has_cation_pH74": float(pos >= 0.5),
        "has_anion_pH74": float(neg >= 0.5),
        "is_zwitterion": float(pos >= 0.5 and neg >= 0.5),
        "LogD_pH74": logp + math.log10(f_neutral),
    }


def _thermo_block(mol: Chem.Mol) -> dict[str, float]:
    """Generic ΔG_bind drivers: size, lipophilicity, polarity, H-bonding, shape."""
    return {
        "MolWt": Descriptors.MolWt(mol),
        "MolLogP": Crippen.MolLogP(mol),          # lipophilicity (hydrophobic burial)
        "MolMR": Crippen.MolMR(mol),              # molar refractivity ~ polarizability
        "TPSA": rdMolDescriptors.CalcTPSA(mol),   # polar surface / desolvation cost
        "LabuteASA": rdMolDescriptors.CalcLabuteASA(mol),
        "NumHDonors": Lipinski.NumHDonors(mol),
        "NumHAcceptors": Lipinski.NumHAcceptors(mol),
        "NumRotatableBonds": Descriptors.NumRotatableBonds(mol),  # conformational entropy
        "FractionCSP3": rdMolDescriptors.CalcFractionCSP3(mol),
        "NumHeavyAtoms": mol.GetNumHeavyAtoms(),
        "NumRings": rdMolDescriptors.CalcNumRings(mol),
        "NumAromaticRings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "NumAliphaticRings": rdMolDescriptors.CalcNumAliphaticRings(mol),
    }


def _aromatic_planarity_block(mol: Chem.Mol) -> dict[str, float]:
    """Flatness / polyaromatic character — CYP1A2's narrow planar slot."""
    n_heavy = max(mol.GetNumHeavyAtoms(), 1)
    n_arom_atoms = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
    n_sp2 = sum(1 for a in mol.GetAtoms() if a.GetHybridization() == Chem.HybridizationType.SP2)
    return {
        "FracAromaticAtoms": n_arom_atoms / n_heavy,
        "FracSP2": n_sp2 / n_heavy,
        "NumAromaticCarbocycles": rdMolDescriptors.CalcNumAromaticCarbocycles(mol),
        "NumAromaticHeterocycles": rdMolDescriptors.CalcNumAromaticHeterocycles(mol),
    }


def _motif_block(mol: Chem.Mol) -> dict[str, float]:
    """Counts of physically/mechanistically meaningful substructures."""
    out = {name: len(mol.GetSubstructMatches(patt)) for name, patt in _COMPILED.items()}
    # aggregate summaries the models can lean on directly
    out["n_heme_coord_total"] = sum(v for k, v in out.items() if k.startswith("heme_"))
    out["n_basic_total"] = sum(v for k, v in out.items() if k.startswith("basic_"))
    out["n_acid_total"] = sum(v for k, v in out.items() if k.startswith("acid_"))
    out["n_tdi_alert_total"] = sum(v for k, v in out.items() if k.startswith("tdi_"))
    out["FormalCharge"] = Chem.GetFormalCharge(mol)
    return out


_GEOM3D_KEYS = [
    "PBF", "NPR1", "NPR2", "Asphericity", "Eccentricity",
    "RadiusOfGyration", "SpherocityIndex", "InertialShapeFactor",
    "basicN_aromatic_min_dist",
]


def _geometry_block_3d(mol: Chem.Mol, seed: int = 0xC0FFEE) -> dict[str, float]:
    """Conformer-derived shape: planarity (PBF, disc-like NPR) for CYP1A2's flat
    slot, and the basic-N ↔ aromatic-ring distance that defines the CYP2D6
    pharmacophore. Returns NaNs if embedding fails."""
    nan = {k: np.nan for k in _GEOM3D_KEYS}
    mh = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mh, params) != 0:
        # retry with random coords for awkward systems
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(mh, params) != 0:
            return nan
    try:
        AllChem.MMFFOptimizeMolecule(mh, maxIters=200)
    except Exception:
        pass
    out = {
        "PBF": rdMolDescriptors.CalcPBF(mh),
        "NPR1": Descriptors3D.NPR1(mh),
        "NPR2": Descriptors3D.NPR2(mh),
        "Asphericity": Descriptors3D.Asphericity(mh),
        "Eccentricity": Descriptors3D.Eccentricity(mh),
        "RadiusOfGyration": Descriptors3D.RadiusOfGyration(mh),
        "SpherocityIndex": Descriptors3D.SpherocityIndex(mh),
        "InertialShapeFactor": Descriptors3D.InertialShapeFactor(mh),
        "basicN_aromatic_min_dist": _basicN_aromatic_dist(mh),
    }
    return out


def _basicN_aromatic_dist(mh: Chem.Mol) -> float:
    """Min distance (Å) from a basic amine N to any aromatic ring centroid — the
    geometric core of the CYP2D6 pharmacophore. NaN if either feature is absent."""
    conf = mh.GetConformer()
    basic = mh.GetSubstructMatches(_BASIC_C["prim_amine"][0]) \
        + mh.GetSubstructMatches(_BASIC_C["sec_amine"][0]) \
        + mh.GetSubstructMatches(_BASIC_C["tert_amine"][0])
    n_idx = [t[0] for t in basic]
    ri = mh.GetRingInfo()
    centroids = []
    for ring in ri.AtomRings():
        if all(mh.GetAtomWithIdx(a).GetIsAromatic() for a in ring):
            pts = np.array([list(conf.GetAtomPosition(a)) for a in ring])
            centroids.append(pts.mean(axis=0))
    if not n_idx or not centroids:
        return np.nan
    npos = np.array([list(conf.GetAtomPosition(i)) for i in n_idx])
    cen = np.array(centroids)
    d = np.sqrt(((npos[:, None, :] - cen[None, :, :]) ** 2).sum(-1))
    return float(d.min())


def featurize_mol(mol: Chem.Mol, include_3d: bool = False) -> dict[str, float]:
    feats = {
        **_thermo_block(mol),
        **_aromatic_planarity_block(mol),
        **_motif_block(mol),
        **_protonation_block(mol),
    }
    if include_3d:
        feats.update(_geometry_block_3d(mol))
    return feats


def featurize_smiles(smiles_list, include_3d: bool = False,
                     cache_path: str | Path | None = None) -> pd.DataFrame:
    """Featurize an iterable of SMILES → DataFrame (aligned to input order).

    Rows whose SMILES fail to parse become all-NaN; the boolean parse mask is in
    ``df.attrs['valid']``. When ``include_3d`` is set, per-molecule 3D descriptors
    are computed (a conformer embedding, ~0.05-0.2 s each) and, if ``cache_path`` is
    given, memoized on disk keyed by canonical SMILES so re-runs are instant.
    """
    smiles_list = list(smiles_list)
    cache: dict[str, dict] = {}
    cache_path = Path(cache_path) if cache_path else None
    if cache_path and cache_path.exists():
        cdf = pd.read_pickle(cache_path)  # pickle: no parquet-engine dependency
        cache = {row["_smi"]: row.drop("_smi").to_dict() for _, row in cdf.iterrows()}

    records, valid, new_rows = [], [], {}
    template = None
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if mol is None:
            records.append(None)
            valid.append(False)
            continue
        key = Chem.MolToSmiles(mol)
        if key in cache:
            feats = cache[key]
        else:
            feats = featurize_mol(mol, include_3d=include_3d)
            if cache_path is not None:
                new_rows[key] = feats
                cache[key] = feats
        template = template or feats
        records.append(feats)
        valid.append(True)

    if template is None:
        raise ValueError("No valid molecules to featurize.")
    if cache_path is not None and new_rows:
        merged = {**cache}
        out_df = pd.DataFrame([{"_smi": k, **v} for k, v in merged.items()])
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_pickle(cache_path)

    filled = [r if r is not None else {k: np.nan for k in template} for r in records]
    df = pd.DataFrame(filled).reset_index(drop=True)
    df.attrs["valid"] = np.array(valid)
    return df


def murcko_scaffold(smiles: str) -> str:
    """Bemis-Murcko scaffold SMILES (empty string if parsing fails). Used to build
    scaffold-disjoint CV folds so internal validation estimates generalization to
    novel chemotypes rather than to the public-leaderboard split."""
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return ""
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except Exception:
        return ""

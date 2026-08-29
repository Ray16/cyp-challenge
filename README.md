# OpenADMET CYP Inhibition Blind Challenge — physics-informed featurization + D-MPNN

Predicting cytochrome-P450 **direct inhibition** — pIC50 for **CYP1A2 / CYP2C9 / CYP2D6 /
CYP3A4** across 750 blinded compounds — for the
[OpenADMET CYP Inhibition Blind Challenge](https://huggingface.co/spaces/openadmet/cyp-challenge)
([announcement](https://openadmet.ghost.io/announcing-openadmets-cyp-inhibition-blind-challenge/)).
Scored by macro-averaged **ST-RAE** (soft-threshold relative absolute error).

---

## Method

### 0. Framing: pIC50 is a binding free energy, and rank ≠ placement

Under fixed assay conditions the Cheng–Prusoff relation gives `pIC50 ≈ pKi + const`, and
`pKi = −ΔG_bind / (2.303·RT)`. So predicting pIC50 is predicting a protein–ligand
**binding free energy**, which motivates two design decisions that run through everything:

1. **ΔG_bind is physically decomposable** (hydrophobic burial, electrostatics/salt bridges,
   H-bonds, π-stacking, heme coordination, desolvation, entropy) — so we engineer
   **interpretable, pocket-specific descriptors** rather than rely only on generic fingerprints.
2. A prediction has **two separable parts**:
   - **Rank** — the ordering of compounds by potency. This is what the *model* learns.
   - **Placement** — the absolute position on the pIC50 axis (mean and spread). This is a
     *calibration*, not the model, and under ST-RAE it dominates the score.

   Conflating the two is the classic trap here (see *Lessons*); the pipeline handles them
   with different machinery — a learned model for rank, an explicit affine transform for
   placement.

### 1. The metric drives the objective

The official metric (`src/cyp_challenge/vendored/official_scoring.py`, copied verbatim from
OpenADMET so it matches the leaderboard) is a **soft-thresholded relative absolute error**:

```
             Σ_i  [ max(0, ŷ_i − hi_i) + max(0, lo_i − ŷ_i) ]
ST-RAE  =   ────────────────────────────────────────────────────
             Σ_i  [ max(0, ȳ  − hi_i) + max(0, lo_i − ȳ ) ]
```

where `[lo_i, hi_i]` is compound *i*'s experimental credible interval and `ȳ = mean(y)`.
The numerator zeroes any prediction that lands inside a compound's CI; the **denominator
is the same soft-thresholding applied to the constant mean-predictor**. Hence **ST-RAE = 1.0
means "no better than predicting the global mean," and > 1.0 is worse.** The denominator's
mass sits on compounds far from the mean (the potent actives), so the metric rewards
accuracy on the actives and penalizes confident deviation on easy near-mean compounds.

We therefore train models to maximize **rank** (out-of-fold Pearson) and fix placement
separately with recalibration — a cleaner, more stable target than optimizing ST-RAE directly.

### 2. Physics-informed featurization (`src/cyp_challenge/features.py`)

Descriptors are grouped by the physical term they proxy:

| block | features | physical rationale |
|---|---|---|
| **Thermodynamic** | MolWt, MolLogP, molar refractivity, TPSA, LabuteASA, H-bond donors/acceptors, rotatable bonds, FractionCSP3, ring counts | generic ΔG_bind drivers: hydrophobic burial, polarizability, desolvation, conformational entropy |
| **Protonation @ pH 7.4** | per-group Henderson–Hasselbalch charge, net charge, strongest-base pKa, **logD** | CYP2D6 binds a *protonated* basic amine via the Asp301 salt bridge; CYP2C9 (Arg108) favors anions. The correct feature is the **charged species**, not a raw atom count |
| **Heme coordination** | accessible sp²-N lone pairs (imidazole / triazole / pyridine), nitrile counts | Type II inhibition (direct N→Fe coordination) — the strongest single potency motif; why azoles are pan-CYP inhibitors |
| **3D shape** | plane-of-best-fit, normalized principal moments (NPR1/2), asphericity, radius of gyration, basic-N ↔ aromatic-ring distance | CYP1A2's narrow, flat active site selects planar polyaromatics; the N↔aromatic distance is the CYP2D6 pharmacophore geometry |
| **Bioactivation alerts** | SMARTS toxicophores (furan, thiophene, terminal alkyne, methylenedioxyphenyl, aniline, …) | precursors of reactive metabolites → mechanism-based (time-dependent) inhibition |

**Protonation model.** For each ionizable group with representative pKa, the fraction
ionized at pH 7.4 follows Henderson–Hasselbalch — bases `f = 1/(1 + 10^(pH−pKa))`, acids
`f = 1/(1 + 10^(pKa−pH))`. Net charge is `Σ_bases f − Σ_acids f`, and
`logD = logP + log₁₀(f_neutral)`. (Swap in a dedicated pKa predictor for microstate rigor.)

**3D descriptors** come from a single ETKDG conformer (MMFF-optimized), cached to disk so
re-runs are instant.

### 3. Proteochemometric target descriptors (`src/cyp_challenge/targets.py`)

To model each isoform on its own physics *and* generalize to an unseen target, each CYP
pocket is described by transferable properties (volume, hydrophobicity, **net charge**,
planarity preference, H-bond capacity, heme accessibility), and crossed with the ligand
features into **interaction terms** — e.g. salt-bridge complementarity
`−(q_ligand · q_pocket)`, hydrophobic `logP · hydrophobicity_pocket`, planar fit, heme
coordination, steric `MW / volume`. One pooled model over
`[ligand ‖ pocket ‖ interactions]` then predicts every isoform and extends to a new pocket
by supplying its descriptors. `run_pcm.py` tests this with **leave-one-CYP-out** (transfer
to a target never trained on).

### 4. Model — a multi-task D-MPNN with physics aux features (`src/cyp_challenge/…`, `scripts/run_dmpnn*.py`)

A learned graph representation supplies the rank; our physics descriptors ride along as
auxiliary molecule-level features. Implemented from scratch in PyTorch (no `chemprop`
dependency).

**Directed message passing** over the molecular graph. Atom features `x_u` (element,
degree, charge, #H, hybridization, aromaticity, ring, mass) and bond features `e_{uv}`
(type, conjugation, ring). For each *directed* bond `u→v`:

```
h⁰_{uv} = ReLU(W_i · [x_u ‖ e_{uv}])
m_{uv}  = Σ_{w ∈ N(u)\{v}} h_{wu}                 (messages in, minus the reverse edge)
h_{uv}  = ReLU(h⁰_{uv} + W_h · m_{uv})            (T = 4 steps, dropout)
```

Atom readout and molecule embedding:

```
m_u   = Σ_{w ∈ N(u)} h_{wu}
h_u   = ReLU(W_o · [x_u ‖ m_u])
g     = mean_u h_u          (hidden 300)
```

The molecule vector `g` is concatenated with the standardized **physics descriptor
vector** `z` (the `chemprop-mt-aux` recipe) and passed to a feed-forward trunk with **8
multi-task heads**: 4 direct-inhibition pIC50 endpoints + 4 auxiliary single-concentration
log2-fold-change endpoints. Training uses a **masked squared-error loss** over the sparse
label matrix (each compound contributes only its measured tasks), so weak-signal CYP2D6
borrows representation from the correlated isoforms and from the dense screening data.
Adam, and a **4-model ensemble** for the final predictions.

### 5. Placement — affine recalibration (`scripts/run_recalibrate.py`)

A model's predictions have the right order but not the right position. Because
`R² = 2ρk − k² − b²` (with `k = σ_pred/σ_true`, `b` = mean offset in units of `σ_true`,
`ρ` = rank correlation), an **affine transform sets center and spread without moving any
compound's rank**:

```
ŷ'_i = μ_blind + z(ŷ_i) · (ρ · σ_blind)          (z = standardized prediction)
```

This is the R²-optimal placement (`k = ρ, b = 0`): a squared-error model *should* be
narrower than the truth by exactly its correlation. Center/spread of the blind
distribution are supplied per isoform; the transform is rank-preserving, so Spearman and
Kendall are unchanged by construction.

### 6. Validation

**Scaffold-disjoint CV** (`src/cyp_challenge/splits.py`): all molecules sharing a
Bemis–Murcko scaffold fall in the same fold, so out-of-fold scores estimate generalization
to novel chemotypes rather than to memorized scaffolds.

---

## Results (scaffold-CV, official vendored ST-RAE; metric = out-of-fold Pearson / rank)

| model | CYP1A2 | CYP2C9 | CYP2D6 | CYP3A4 | macro |
|---|---|---|---|---|---|
| GBM (physics + Morgan) | 0.48 | 0.60 | 0.38 | 0.74 | 0.55 |
| multi-task MLP | 0.50 | 0.64 | 0.39 | 0.77 | 0.57 |
| **D-MPNN (+ physics aux)** | **0.56** | **0.66** | **0.38** | **0.79** | **0.60** |

> ⚠️ **Honest caveat.** Scaffold-OOF Pearson *understates* blind Pearson (the split is
> harder than the blind half). Applying published OOF→blind ratios suggests our D-MPNN's
> blind rank (~0.74 / 0.81 / 0.63 / 0.85) is close to the frontier — **but that is an
> estimate resting on another team's ratios and solved blind moments, not a measured
> result.** The leaderboard is the only arbiter; each submission is treated as an
> experiment that *tests* the estimate.

### Lessons documented because they were expensive
- **The ST-RAE denominator is the *soft-thresholded mean baseline*.** A hand-reconstruction
  with a plain-deviation denominator made internal CV look ~40% better than reality, and
  our first real submission landed near the bottom. We now delegate to the vendored
  official scorer.
- **The train→test gap was a *placement/level shift*, not novel chemistry or fixable
  compression** — all three were falsified with diagnostics (`scripts/run_decompress.py`):
  the test is not chemically OOD, decompression *worsens* the score, and CI-width explains
  little. The remedy is recalibration, not more features.

---

## Repo layout
```
src/cyp_challenge/
  features.py     physics-informed featurizer (thermo, protonation, heme, 3D, alerts)
  targets.py      CYP pocket descriptors + ligand×pocket interaction terms (PCM)
  metrics.py      ST-RAE (delegates to the vendored official scorer)
  splits.py       scaffold-disjoint CV folds
  validate.py     local pre-flight submission checks
  vendored/       OpenADMET's official scorer, verbatim (Apache-2.0)
scripts/
  run_baseline.py     per-isoform GBM baseline
  run_ablation.py     physics vs fingerprints ablation
  run_pcm.py          proteochemometric leave-one-CYP-out (target transfer)
  run_mtl.py          multi-task MLP
  run_dmpnn.py        D-MPNN multi-task, OOF rank
  run_dmpnn_submit.py D-MPNN ensemble + recalibration → submission
  run_recalibrate.py  affine placement recalibration
  run_decompress.py   train→test gap diagnostics
```

## Reproduce
```bash
pip install -r requirements.txt        # pandas numpy scipy scikit-learn rdkit torch huggingface_hub

# 1. Download the challenge data into data/raw/ (not redistributed here):
python - <<'PY'
import pandas as pd, pathlib; pathlib.Path("data/raw").mkdir(parents=True, exist_ok=True)
base = "hf://datasets/openadmet/cyp-challenge-train-test/"
for f in ["cyp-challenge-TRAIN_inhibition","cyp-challenge-TEST-BLINDED",
          "cyp-challenge-TRAIN_TDI","cyp-challenge-TRAIN_Emax",
          "cyp-challenge-single-concentration-TRAIN"]:
    pd.read_csv(base+f+".csv").to_csv(f"data/raw/{f}.csv", index=False)
PY

# 2. Run (GPU used automatically if available)
python scripts/run_ablation.py --with-3d    # physics vs fingerprints
python scripts/run_dmpnn.py                 # D-MPNN OOF rank
python scripts/run_dmpnn_submit.py          # build a recalibrated submission
```

## Attribution & license
- Our code: MIT (`LICENSE`).
- `src/cyp_challenge/vendored/official_scoring.py` — OpenADMET's official scorer, verbatim
  from [CYP-Challenge-Tutorial](https://github.com/OpenADMET/CYP-Challenge-Tutorial), Apache-2.0.
- The blind-distribution moments and OOF→blind ratios used in recalibration are from the
  public [SuperCowPowers/workbench](https://github.com/SuperCowPowers/workbench) OpenADMET
  CYP pipeline (used as a prior; see caveat above); their `cyp_recalibrate.py` documents the
  order-vs-placement decomposition this builds on.
- Data © OpenADMET (`openadmet/cyp-challenge-train-test`, Apache-2.0) — download it yourself;
  it is not redistributed here.

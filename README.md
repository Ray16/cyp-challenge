# OpenADMET CYP Inhibition Blind Challenge — physics-informed + D-MPNN

Predicting cytochrome-P450 **direct inhibition** (pIC50 for CYP1A2 / CYP2C9 / CYP2D6 /
CYP3A4) for 750 blinded compounds, for the
[OpenADMET CYP Inhibition Blind Challenge](https://huggingface.co/spaces/openadmet/cyp-challenge)
([announcement](https://openadmet.ghost.io/announcing-openadmets-cyp-inhibition-blind-challenge/)).
Scored by macro-averaged **ST-RAE** (soft-threshold relative absolute error).

## Approach in one idea

pIC50 is, to first order, a protein–ligand **binding free energy** (ΔG ≈ −2.303·RT·pKi),
so we (1) build **physics-informed, per-pocket descriptors** instead of opaque
fingerprints, and (2) treat prediction as **two separable problems**:

- **Rank** — the ordering of compounds by potency. This is the *model*.
- **Placement** — where predictions sit on the pIC50 axis (mean & spread). This is *not*
  the model; it's a calibration, and under ST-RAE it matters enormously.

The final pipeline combines a learned representation for rank with an explicit
recalibration for placement:

1. **D-MPNN** (ChemProp-style directed message passing, implemented from scratch in
   PyTorch — no `chemprop` dependency) with our **physics descriptors as auxiliary
   molecule features**, and **multi-task heads** (4 pIC50 + 4 single-concentration
   log2-fold-change) under a masked loss over the sparse label matrix.
2. **Affine recalibration** of each isoform's predictions onto the blind test
   distribution (rank-preserving; fixes center and spread).

### Physics descriptors (`src/cyp_challenge/features.py`)
- **Thermodynamic drivers** — logP/logD, molar refractivity, TPSA, size, flexibility.
- **Protonation at pH 7.4** — Henderson–Hasselbalch pKa model → basic/acidic charge
  state, strongest-base pKa, logD. (Drives CYP2D6's Asp301 salt bridge, CYP2C9's Arg108.)
- **Heme coordination** — accessible sp2-N lone pairs (imidazole/triazole/pyridine) →
  Type II inhibition, the strongest single potency motif (why azoles are pan-CYP inhibitors).
- **3D planarity** — PBF / normalized principal moments (CYP1A2's flat active site).
- **Bioactivation alerts** — SMARTS toxicophores, for the (separate) TDI track.

## Results (scaffold-disjoint CV, official vendored ST-RAE)

Rank quality (out-of-fold Pearson), the metric we optimize for the model since
placement is handled separately:

| model | CYP1A2 | CYP2C9 | CYP2D6 | CYP3A4 | macro |
|---|---|---|---|---|---|
| GBM (physics + Morgan) | 0.48 | 0.60 | 0.38 | 0.74 | 0.55 |
| multi-task MLP | 0.50 | 0.64 | 0.39 | 0.77 | 0.57 |
| **D-MPNN (+ physics aux)** | **0.56** | **0.66** | **0.38** | **0.79** | **0.60** |

> ⚠️ **Honest caveats.** Scaffold-OOF Pearson *understates* blind Pearson (the split is
> harder than the blind half). Applying published OOF→blind ratios suggests our D-MPNN's
> blind rank (~0.74/0.81/0.63/0.85) is close to the frontier — **but that is an estimate
> resting on another team's ratios and solved blind moments, not a measured result.** The
> leaderboard is the only arbiter, and this repo treats each submission as an experiment
> that *tests* the estimate.

### Lessons documented because they were expensive
- **The official ST-RAE denominator is the *soft-thresholded mean baseline*** — so
  ST-RAE = 1.0 means "as good as predicting the global mean," and > 1.0 is worse. A
  hand-reconstruction with a plain-deviation denominator made internal CV look ~40%
  better than reality (our first real submission landed near the bottom). We now vendor
  and delegate to the official scorer (`src/cyp_challenge/vendored/`).
- **The train→test gap was a *placement/level shift*, not novel chemistry or fixable
  compression** — all three were falsified with diagnostics (`scripts/run_decompress.py`):
  the test is not chemically OOD, decompression *worsens* the score, and CI-width
  explains little. The fix is recalibration, not more features.

## Repo layout
```
src/cyp_challenge/       # pipeline package
  features.py            #   physics-informed featurizer (+ 3D, protonation, motifs)
  targets.py             #   CYP pocket descriptors + ligand×pocket interaction terms (PCM)
  metrics.py             #   ST-RAE (delegates to vendored official scorer)
  splits.py              #   scaffold-disjoint CV folds
  vendored/              #   official scoring, copied verbatim (Apache-2.0)
scripts/                 # runnable experiments (see header of each)
  run_baseline.py        #   per-isoform GBM baseline
  run_ablation.py        #   physics vs fingerprints ablation
  run_pcm.py             #   proteochemometric leave-one-CYP-out (target transfer)
  run_decompress.py      #   train→test gap diagnostics
  run_dmpnn.py           #   D-MPNN multi-task, OOF rank
  run_dmpnn_submit.py    #   D-MPNN ensemble + recalibration → submission
  run_recalibrate.py     #   affine placement recalibration
src/cyp_challenge/validate.py   # local pre-flight submission checks
```

## Reproduce
```bash
pip install -r requirements.txt            # pandas, numpy, scikit-learn, rdkit, torch, scipy

# 1. Download the challenge data into data/raw/ (NOT redistributed here):
python - <<'PY'
import pandas as pd, pathlib; pathlib.Path("data/raw").mkdir(parents=True, exist_ok=True)
base = "hf://datasets/openadmet/cyp-challenge-train-test/"
for f in ["cyp-challenge-TRAIN_inhibition","cyp-challenge-TEST-BLINDED",
          "cyp-challenge-TRAIN_TDI","cyp-challenge-TRAIN_Emax",
          "cyp-challenge-single-concentration-TRAIN"]:
    pd.read_csv(base+f+".csv").to_csv(f"data/raw/{f}.csv", index=False)
PY

# 2. Run experiments (GPU used automatically if available)
python scripts/run_ablation.py --with-3d      # physics vs fingerprints
python scripts/run_dmpnn.py                   # D-MPNN OOF rank
python scripts/run_dmpnn_submit.py            # build a recalibrated submission
```

## Attribution & license
- Our code: MIT (see `LICENSE`).
- `src/cyp_challenge/vendored/official_scoring.py` — OpenADMET's official scorer,
  vendored verbatim from
  [CYP-Challenge-Tutorial](https://github.com/OpenADMET/CYP-Challenge-Tutorial), Apache-2.0.
- Blind-distribution moments and OOF→blind ratios used by `run_recalibrate.py` /
  `run_dmpnn_submit.py` are from the public
  [SuperCowPowers/workbench](https://github.com/SuperCowPowers/workbench) OpenADMET CYP
  pipeline (used as a prior; see caveats above). Their `cyp_recalibrate.py` documents the
  order-vs-placement decomposition this repo builds on.
- Data © OpenADMET (`openadmet/cyp-challenge-train-test`, Apache-2.0); download it
  yourself — it is not redistributed here.

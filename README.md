# tdc-admet-bench

Feature selection and hyperparameter optimization for molecular property prediction on the [TDC ADMET Benchmark](https://tdcommons.ai/benchmark/admet_group/overview/).

## Our Approach

Fingerprint-based ML models (LightGBM, XGBoost, CatBoost, RF, SVM) with automated feature selection to find optimal fingerprint/descriptor combinations per ADMET task.

1. **Molecular featurization** — 21 fingerprint types (ECFP, FCFP, MACCS, Avalon, etc.) + RDKit/Mordred descriptors via [molfeat](https://molfeat.datamol.io/)
2. **Sequential Feature Selection (SFS)** — greedy forward/backward selection with feature groups (each fingerprint type is an atomic unit), 5×5 repeated cross-validation
3. **Optuna feature selection** — Bayesian optimization over feature group combinations
4. **Evaluation** — TDC official multi-seed protocol (5 seeds) with leaderboard ranking

## External Baselines

Published scores from the [TDC ADMET Leaderboard](https://tdcommons.ai/benchmark/admet_group/overview/) (41 models across 22 benchmarks) are stored in `data/tdc_admet_leaderboard.json` and used as reference baselines for ranking our models.

## Installation

```bash
conda env create -f environment.yml
conda activate tdc-admet-bench
```

## Quick Start

```bash
# Sequential feature selection on a benchmark
python scripts/run_sfs.py --benchmark caco2_wang --model lgb --k-features 10

# Optuna-based feature selection
python scripts/run_optuna.py --benchmark caco2_wang --model lgb --n-trials 100

# Evaluate a feature combination on all 22 benchmarks
python scripts/run_evaluate.py --benchmark all --model lgb --features ecfp,maccs,desc2D
```

## Project Structure

```
tdc_admet_bench/                # Our models and pipeline
├── config.py          # Benchmark metadata, metrics, fingerprint defaults
├── preprocess.py      # SMILES standardization (datamol)
├── features.py        # Fingerprint/descriptor transformers + matrix builder
├── models.py          # Model registry (RF, XGB, LightGBM, CatBoost, SVM)
├── sfs.py             # Sequential Feature Selector (forward/backward/floating)
├── optuna_select.py   # Optuna-based feature selection
└── evaluate.py        # Multi-seed TDC evaluation
scripts/
├── run_sfs.py         # Run SFS on a benchmark
├── run_optuna.py      # Run Optuna feature selection
└── run_evaluate.py    # Evaluate on benchmarks
data/
├── admet_group/                # TDC benchmark datasets (22 tasks)
└── tdc_admet_leaderboard.json  # External: published scores from TDC leaderboard
```

## Supported Models

`rf` (Random Forest), `lgb` (LightGBM), `xgb` (XGBoost), `cat` (CatBoost), `svm` (SVM), `ridge`, `hist` (HistGradientBoosting)

## Supported Fingerprints (21 types)

| Type | Configurable | Size |
|------|-------------|------|
| ecfp, fcfp, avalon, rdkit, topological, atompair, pattern, layered, secfp | length | default 1024 |
| ecfp-count, fcfp-count, rdkit-count, topological-count, atompair-count | length | default 1024 |
| maccs | fixed | 167 |
| erg | fixed | 315 |
| estate | fixed | 79 |
| desc2D | fixed | 223 |
| cats2D | fixed | 189 |
| scaffoldkeys | fixed | 42 |
| skeys | fixed | 42 |
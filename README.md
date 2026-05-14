# tdc-admet-bench

Code and data archive for *Critical Assessment of ML models for ADMET Prediction in TDC leaderboards* (Koleiev et al., Receptor.AI).

The repo holds two tiers:

1. **In-house models** — Mol2Vec + RDKit/Mordred + LightGBM with sequential feature selection (SFS) and Optuna HPO. Honest and deliberately-overfit variants from manuscript Section 3.
2. **Third-party re-evaluation wrappers** — fresh, pinned wrappers around the three TDC leaderboard models that passed all four validation stages in our study: MapLight, MapLight+GNN, CaliciBoost. One wrapper per model, each reproducing the relevant row of Table 3.

## Data

The TDC ADMET benchmark snapshot used throughout the paper (downloaded 2026-03-24 via PyTDC 0.3.8) lives at `data/admet_group/` — 22 endpoints × `{train_val.csv, test.csv}`. The same snapshot is archived on Zenodo: **DOI TBD** (replace once deposit is finalized).

External published TDC leaderboard scores used as reference baselines are stored in `data/tdc_admet_leaderboard.json`.

## Scope

| Model | Status in this repo | Reason |
|---|---|---|
| In-house Mol2Vec+LightGBM (honest, overfit) | full train + inference (`scripts/`, `tdc_admet_bench/`) | manuscript Section 3 |
| MapLight (22 endpoints) | full train + inference (`models/maplight/`) | Stage-4 survivor |
| MapLight+GNN (22 endpoints) | full train + inference (`models/maplight_gnn/`) | Stage-4 survivor |
| CaliciBoost (Caco-2 only) | full train + inference (`models/caliciboost/`) | Stage-4 survivor |
| ADMETrix, CFA, SimGCN, ZairaChem, MiniMol, GradientBoost+, XGBoost | not included | eliminated at Stages 1–3; failure modes documented in the manuscript Supporting Information |

## In-house pipeline

Fingerprint-based ML models (LightGBM, XGBoost, CatBoost, RF, SVM) with automated feature selection over 21 fingerprint types (ECFP, FCFP, MACCS, Avalon, …) + RDKit/Mordred descriptors via [molfeat](https://molfeat.datamol.io/).

```bash
conda env create -f environment.yml
conda activate tdc-admet-bench

# Sequential forward selection on a benchmark
python scripts/run_sfs.py --benchmark caco2_wang --model lgb --k-features 10

# Optuna-based feature selection
python scripts/run_optuna.py --benchmark caco2_wang --model lgb --n-trials 100

# Evaluate a fingerprint combination on all 22 benchmarks (multi-seed TDC protocol)
python scripts/run_evaluate.py --benchmark all --model lgb --features ecfp,maccs,desc2D
```

Mol2Vec embeddings come from the proprietary `rai_mol2vec` package. Without it the in-house scripts cannot run end-to-end; the SFS/HPO logic is still useful as a reference implementation, and the other fingerprint+descriptor groups work out of the box.

## Third-party wrappers

Each wrapper is self-contained: vendored upstream source under `upstream/`, our re-evaluation driver under `wrapper/`, pinned conda env, output CSVs under `wrapper/results/` (gitignored — regenerable).

```bash
# MapLight (22 endpoints)
cd models/maplight
conda env create -f wrapper/environment.yml
conda activate maplight-tdc
PYTHONNOUSERSITE=1 python wrapper/run_all.py

# MapLight+GNN (22 endpoints)
cd models/maplight_gnn
PYTHONNOUSERSITE=1 conda env create -f wrapper/environment.yml
conda activate maplight-gnn-tdc
PYTHONNOUSERSITE=1 python wrapper/run_all.py

# CaliciBoost (Caco-2 only)
cd models/caliciboost
conda env create -f wrapper/environment.yml
conda activate caliciboost-tdc
PYTHONNOUSERSITE=1 python wrapper/run_caco2.py
```

`PYTHONNOUSERSITE=1` guards against `~/.local/lib/python3.x/site-packages` shadowing the conda env; drop it if your machine has no user-site packages. Per-model `README.md` files cover provenance, the exact upstream commits, and any minor deviations (e.g. CaliciBoost's `gpu_hist` → `hist` switch for CPU reproducibility).

## TDC split

All re-evaluations use the official scaffold split returned by `tdc.benchmark_group.admet_group`, unmodified. 5 seeds per endpoint per the TDC multi-seed protocol.

## Project structure

```
tdc-admet-bench/
├── environment.yml                  # top-level env for in-house pipeline
├── tdc_admet_bench/                 # in-house pipeline library
│   ├── config.py                    #   benchmark metadata, metrics, fingerprint defaults
│   ├── preprocess.py                #   SMILES standardization (datamol)
│   ├── features.py                  #   fingerprint/descriptor transformers
│   ├── models.py                    #   model registry (RF, XGB, LightGBM, CatBoost, SVM)
│   ├── sfs.py                       #   sequential feature selector
│   ├── optuna_select.py             #   Optuna-based feature selection
│   └── evaluate.py                  #   multi-seed TDC evaluation
├── scripts/                         # in-house CLIs
│   ├── run_sfs.py
│   ├── run_optuna.py
│   └── run_evaluate.py
├── models/                          # third-party re-evaluation wrappers
│   ├── maplight/
│   ├── maplight_gnn/
│   └── caliciboost/
└── data/
    ├── admet_group/                 # TDC snapshot (22 endpoints, 2026-03-24)
    └── tdc_admet_leaderboard.json   # published TDC leaderboard scores
```

## Supported in-house models and fingerprints

`rf`, `lgb`, `xgb`, `cat`, `svm`, `ridge`, `hist`.

21 fingerprint types via molfeat: `ecfp`, `fcfp`, `avalon`, `rdkit`, `topological`, `atompair`, `pattern`, `layered`, `secfp` (+ `-count` variants for the first five), `maccs`, `erg`, `estate`, `desc2D`, `cats2D`, `scaffoldkeys`, `skeys`.

## Third-party licensing

Each `models/*/upstream/` directory carries the upstream model's own MIT license file and a `PROVENANCE.md` pinning the source commit. The wrapper code under `models/*/wrapper/` is original to this repository.

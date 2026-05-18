# tdc-admet-bench

Code and data for **"Critical Assessment of ML Models for ADMET Prediction in TDC Leaderboards"** (Koleiev et al., Receptor.AI, 2026).

> Paper: <https://www.biorxiv.org/content/10.64898/2026.02.26.708193v1.full>
> Data archive (Zenodo): <https://doi.org/10.5281/zenodo.20180944>

## TL;DR

The [TDC ADMET leaderboard](https://tdcommons.ai/benchmark/admet_group/overview/) is a public benchmark for predicting drug-like properties (absorption, distribution, metabolism, excretion, toxicity). We screened the **top 3 leaderboard models on each of the 22 endpoints — 10 distinct models in total** — and tried to actually reproduce them. Most could not be reproduced — either the environment didn't build, the training set leaked into the test set, the published validation split included TDC test molecules, or the reported numbers couldn't be recovered. **Only 3 out of 10 survived all four checks**: MapLight, MapLight+GNN, and CaliciBoost (Caco-2 only).

This repository contains everything needed to repeat that assessment, plus our own re-evaluation pipeline that uses standard fingerprints and gradient-boosted trees as a baseline.

## Quick start

```bash
# 1. Create env
conda env create -f environment.yml
conda activate tdc-admet-bench

# 2. Run our pipeline on one TDC endpoint (smoke test)
python scripts/run_evaluate.py --benchmark caco2_wang --model lgb --features ecfp,maccs
```

For full paper reproduction, see [Reproducing the paper](#reproducing-the-paper) below.

## What's in this repo

This is two things bundled together:

1. **Our re-evaluation pipeline.** A LightGBM/XGBoost/CatBoost framework that takes a TDC benchmark, computes molecular fingerprints and descriptors (via [molfeat](https://molfeat.datamol.io/)), runs feature selection, and reports leaderboard-style scores. Used in Section 3 of the paper to produce both an "honest" baseline (trained only on the official TDC train split) and a deliberately "overfit" variant (tuned against the public test set — for illustrating how easy it is to game the leaderboard).
2. **Wrappers around the three third-party models that passed validation.** Each wrapper vendors the upstream code at a pinned commit, ships its own conda environment, and provides one command to run the model across the relevant TDC endpoints.

### Which models are in scope

The paper assesses the 10 distinct models that appeared in the top 3 on at least one of the 22 TDC endpoints. Only 3 are included here as runnable code; the others were eliminated at earlier stages and their failure modes are documented in the paper.

| Model | In this repo? | Why |
|---|---|---|
| Our LightGBM/XGBoost/CatBoost pipeline on public fingerprints (honest and overfit) | yes | Section 3 of the paper. The Mol2Vec rows from Section 3 rely on an internal Receptor.AI package that is not public and cannot be reproduced from this repo. |
| MapLight, all 22 endpoints | yes | passed all validation stages |
| MapLight+GNN, all 22 endpoints | yes | passed all validation stages |
| CaliciBoost, Caco-2 only | yes | passed all validation stages |
| ADMETrix, CFA, SimGCN, ZairaChem | no | Stage 1 — environment failed to build |
| MiniMol | no | Stage 2 — pretraining data leaked into TDC test set |
| GradientBoost+, XGBoost | no | Stage 3 — author's validation split included TDC test molecules |

## Data

We used the TDC ADMET benchmark group as it stood on **2026-03-24**, downloaded with PyTDC 0.3.8. It is checked into `data/admet_group/` (22 endpoints, each with `train_val.csv` and `test.csv`) and also archived on Zenodo: <https://doi.org/10.5281/zenodo.20180944>.

The published leaderboard scores we compared against are in `data/tdc_admet_leaderboard.json`.

All evaluations use the official TDC scaffold split untouched, with the standard 5-seed protocol from `tdc.benchmark_group.admet_group`.

## Reproducing the paper

| Paper artefact | Command |
|---|---|
| **Section 3 — our own models, honest and overfit variants (one run produces both)** | `python scripts/run_optuna.py --benchmark <endpoint> --model lgb --n-trials 100` |
| **Section 4 — MapLight, all 22 endpoints** | `cd models/maplight && conda env create -f wrapper/environment.yml && conda activate maplight-tdc && PYTHONNOUSERSITE=1 python wrapper/run_all.py` |
| **Section 4 — MapLight+GNN, all 22 endpoints** | `cd models/maplight_gnn && PYTHONNOUSERSITE=1 conda env create -f wrapper/environment.yml && conda activate maplight-gnn-tdc && PYTHONNOUSERSITE=1 python wrapper/run_all.py` |
| **Section 4 — CaliciBoost on Caco-2** | `cd models/caliciboost && conda env create -f wrapper/environment.yml && conda activate caliciboost-tdc && PYTHONNOUSERSITE=1 python wrapper/run_caco2.py` |

> **About the Section 3 run.** `run_optuna.py` takes a single benchmark name (the 22 endpoint names are listed in `tdc_admet_bench/config.py` under `BENCHMARK_CONFIG`); to sweep all 22, wrap it in a shell loop. Each run produces two reported outcomes side-by-side: the *best* trial (feature subset selected by 5×5 repeated-CV on the training set — the **honest** baseline) and the *oracle* trial (feature subset selected by held-out test score — the **overfit** variant used in the paper to demonstrate how much leaderboard rank can be inflated by selecting on test).

> **Why `PYTHONNOUSERSITE=1`?** It stops Python from picking up packages installed in `~/.local/lib/python3.x/site-packages`, which otherwise leak into the conda env and break the pinned versions. Skip it if your machine has no user-site packages.

Each model's own `models/<name>/README.md` lists the upstream commit and any small patches we applied (for example, switching CaliciBoost from `gpu_hist` to `hist` so it runs reproducibly on CPU).


## Our pipeline in more detail

Models available: `rf`, `lgb`, `xgb`, `cat`, `svm`, `ridge`, `hist` (random forest, LightGBM, XGBoost, CatBoost, SVM, ridge regression, sklearn HistGradientBoosting).

Fingerprints (21 families, all via molfeat): `ecfp`, `fcfp`, `avalon`, `rdkit`, `topological`, `atompair`, `pattern`, `layered`, `secfp`, the `-count` variants of the first five, `maccs`, `erg`, `estate`, `desc2D`, `cats2D`, `scaffoldkeys`, `skeys`.

Two feature-selection strategies are provided:

- **Sequential forward selection (SFS)** — greedy add-one-at-a-time, the classic approach. Fast and interpretable.
- **Optuna-based selection** — treats each fingerprint family as a binary include/exclude choice and uses [Optuna](https://optuna.org/)'s TPE sampler to find the best combination. Model hyperparameters are fixed; only the feature subset is searched. Each run reports two outcomes: the *best* trial (selected by 5×5 repeated-CV score on the training set — the honest variant) and the *oracle* trial (selected by held-out test score — used in the paper to illustrate how much leaderboard rank can be inflated by selecting on test).

```bash
# Sequential forward selection on one benchmark
python scripts/run_sfs.py --benchmark caco2_wang --model lgb --k-features 10

# Optuna-based feature selection
python scripts/run_optuna.py --benchmark caco2_wang --model lgb --n-trials 100

# Evaluate a fixed feature combination across all 22 benchmarks
python scripts/run_evaluate.py --benchmark all --model lgb --features ecfp,maccs,desc2D
```

## Repository layout

```
tdc-admet-bench/
├── environment.yml                  # env for our pipeline
├── tdc_admet_bench/                 # our pipeline as a library
│   ├── config.py                    #   benchmark metadata, metrics, defaults
│   ├── preprocess.py                #   SMILES standardization (datamol)
│   ├── features.py                  #   fingerprint and descriptor transformers
│   ├── models.py                    #   model registry
│   ├── sfs.py                       #   sequential feature selector
│   ├── optuna_select.py             #   Optuna-based feature selection
│   └── evaluate.py                  #   multi-seed TDC evaluation
├── scripts/                         # CLIs for the pipeline
│   ├── run_sfs.py
│   ├── run_optuna.py
│   └── run_evaluate.py
├── models/                          # third-party wrappers
│   ├── maplight/                    #   each has upstream/, wrapper/, README.md
│   ├── maplight_gnn/
│   └── caliciboost/
└── data/
    ├── admet_group/                 # TDC snapshot, 2026-03-24
    └── tdc_admet_leaderboard.json   # published leaderboard scores
```

## Requirements

- Python **3.10** (pinned in `environment.yml`)
- CPU only — no GPU required for any model
- See `environment.yml` for the full dependency list

## License

Each `models/<name>/upstream/` directory keeps the upstream project's original license alongside a `PROVENANCE.md` pinning the source commit.

## Citing

If you use this code, please cite the paper:

```bibtex
@article{koleiev2026admet,
  title   = {Critical Assessment of ML Models for ADMET Prediction in TDC Leaderboards},
  author  = {Koleiev, Ihor and Stratiichuk, Roman and Shevchuk, Nazar and Melnychenko, Mykola and Nyporko, Alex and Todoryshyn, Daniil and Husak, Vladyslav and Starosyla, Sergii and Yesylevskyy, Semen and Nafiiev, Alan},
  year    = {2026},
  doi     = {10.64898/2026.02.26.708193v1},
  url     = {https://www.biorxiv.org/content/10.64898/2026.02.26.708193v1.full}
}
```

If you use the data archive, please also cite the Zenodo deposit:

```bibtex
@dataset{tdc_admet_bench_data_2026,
  title     = {TDC ADMET benchmark snapshot (2026-03-24)},
  author    = {Koleiev, Ihor and others},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20180944},
  url       = {https://doi.org/10.5281/zenodo.20180944}
}
```


## Questions

Open a GitHub issue on this repository — that's the fastest route. The repo is maintained by the Receptor.AI team.

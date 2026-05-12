# CaliciBoost re-evaluation

Reproduces the CaliciBoost row of Table 3 in *Critical Assessment of ML models for ADMET Prediction in TDC leaderboards*. Caco-2 only — CaliciBoost is a single-endpoint submission.

## Layout

- `upstream/` — verbatim copy of [Calici/CaliciBoost](https://github.com/Calici/CaliciBoost) at commit `a11251f`, including `PaDEL_Descriptors.csv` (12.6 MB precomputed PaDEL descriptors). MIT license. See `upstream/PROVENANCE.md`.
- `wrapper/run_caco2.py` — our thin re-eval script. 5 seeds. Swaps upstream `tree_method='gpu_hist'` → `tree_method='hist'` so the run is reproducible on CPU.
- `wrapper/environment.yml` — pinned conda env (XGBoost 1.7.3, scikit-learn 1.2.2, numpy 1.26.4, pandas 2.2.3 per upstream notebook).

## Data

Reads `tdc-admet-bench/data/admet_group/caco2_wang/{train_val,test}.csv` (TDC snapshot 2026-03-24, PyTDC 0.3.8). Same snapshot is on Zenodo (DOI TBD).

## Run

```bash
conda env create -f wrapper/environment.yml
conda activate caliciboost-tdc

PYTHONNOUSERSITE=1 python wrapper/run_caco2.py
```

Wall time on CPU: ~15–30 min for 5 seeds.

## Differences from upstream

The only modification vs. the upstream `submission.ipynb`:
- `tree_method='gpu_hist'` → `tree_method='hist'`. XGBoost's histogram-based training is numerically identical between CPU and GPU paths.

PaDEL descriptors come from the vendored `PaDEL_Descriptors.csv`. `padelpy` is included in the env for any new molecules not in the precomputed file, but for the TDC Caco-2 test set the precomputed CSV covers everything.

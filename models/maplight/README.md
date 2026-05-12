# MapLight re-evaluation

Reproduces the MapLight row of Table 3 in *Critical Assessment of ML models for ADMET Prediction in TDC leaderboards*.

## Layout

- `upstream/` — verbatim copy of [maplightrx/MapLight-TDC](https://github.com/maplightrx/MapLight-TDC) at commit `c249378` (MIT license). See `upstream/PROVENANCE.md`.
- `wrapper/` — our thin re-evaluation script:
  - `run_all.py` — loops the upstream code over all 22 TDC ADMET endpoints, 5 seeds each
  - `environment.yml` — pinned conda env (CatBoost 1.2.8, PyTDC 0.3.8)
  - `results/` — generated on run: per-endpoint prediction CSVs + `summary_metrics.csv`

## Data

The script reads from `tdc-admet-bench/data/admet_group/` (TDC snapshot downloaded 2026-03-24 via PyTDC 0.3.8). TDC's `admet_group(path=...)` loader uses cached files when present, so no network call is made and the snapshot is the single source of truth.

The same snapshot is archived on Zenodo (DOI TBD).

## Run

```bash
conda env create -f wrapper/environment.yml
conda activate maplight-tdc

# All 22 endpoints
PYTHONNOUSERSITE=1 python wrapper/run_all.py

# Single endpoint
PYTHONNOUSERSITE=1 python wrapper/run_all.py --endpoint caco2_wang
```

The `PYTHONNOUSERSITE=1` env var guards against `~/.local/lib/python3.x/site-packages` shadowing the conda env. Drop it if your machine has no user-site packages.

Wall time on a single 16-core CPU: ~2–4 hours for all 22 endpoints.

## TDC split

Per the manuscript Section 1.3 — official scaffold split from `tdc.benchmark_group.admet_group`, no modification. The wrapper calls `group.get(benchmark)` exactly as the upstream submission notebook does.

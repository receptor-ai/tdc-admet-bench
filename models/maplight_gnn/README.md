# MapLight+GNN re-evaluation

Reproduces the MapLight+GNN row of Table 3 in *Critical Assessment of ML models for ADMET Prediction in TDC leaderboards*.

## Layout

- `upstream/` — vendored from [maplightrx/MapLight-TDC](https://github.com/maplightrx/MapLight-TDC) at commit `c249378` (MIT license). See `upstream/PROVENANCE.md`. Files: `maplight_gnn.py`, `submission_gnn.ipynb`, `LICENSE`.
- `wrapper/run_all.py` — our thin re-eval script over all 22 endpoints, 5 seeds each. Structurally identical to the MapLight wrapper; the difference is in `upstream/maplight_gnn.py` which appends a 300-dim GIN supervised-masking embedding to MapLight's hand-crafted features.
- `wrapper/environment.yml` — pinned conda env with the heavier GNN stack (molfeat, dgl, dgllife, pytorch).

## Data

Same pinned snapshot as MapLight: `tdc-admet-bench/data/admet_group/`, TDC 2026-03-24 via PyTDC 0.3.8. Zenodo DOI TBD.

## Run

```bash
# IMPORTANT: PYTHONNOUSERSITE=1 must be set during env build too, otherwise pip
# treats ~/.local packages as already-satisfying molfeat's transitive deps and
# silently skips installing them into the env.
PYTHONNOUSERSITE=1 conda env create -f wrapper/environment.yml
conda activate maplight-gnn-tdc

PYTHONNOUSERSITE=1 python wrapper/run_all.py
PYTHONNOUSERSITE=1 python wrapper/run_all.py --endpoint caco2_wang
```

If you hit `ImportError: ... CXXABI_1.3.15 not found` on matplotlib, the env's libstdc++ is being shadowed by an older system one. Force the env's lib dir first:

```bash
LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH PYTHONNOUSERSITE=1 python wrapper/run_all.py
```

Wall time on a single 16-core CPU: ~3–5 hours for all 22 endpoints (the GIN embedding step adds ~30s per endpoint on top of MapLight's runtime).

# Provenance

These files are vendored verbatim from the upstream CaliciBoost repository.

- **Source**: https://github.com/Calici/CaliciBoost
- **Commit**: `a11251f` (HEAD on `main` at vendoring time)
- **Vendored on**: 2026-05-12
- **License**: MIT (see `LICENSE` in this directory)
- **Reference**: Le HV, Ren W, Kim J, Yun Y, Park YB, Kim YJ, et al. *CaliciBoost: Performance-driven evaluation of molecular representations for caco-2 permeability prediction.* J Cheminformatics 17, 184 (2025). doi:10.1186/s13321-025-01137-7

Note: `PaDEL_Descriptors.csv` (12.6 MB) is the upstream-bundled pre-computed PaDEL descriptor matrix for Caco-2 molecules, originally distributed via HuggingFace (`junhong1222/PaDEL_Descriptors`). Kept here so the run is fully offline against the pinned snapshot.

No modifications to the upstream files. Our wrapper at `../wrapper/run_caco2.py` swaps `tree_method='gpu_hist'` → `tree_method='hist'` so the run is reproducible on CPU; XGBoost histogram-based training is numerically identical between CPU and GPU paths.

# Provenance

These files are vendored verbatim from the upstream MapLight TDC repository (same repo as MapLight, separate submission).

- **Source**: https://github.com/maplightrx/MapLight-TDC
- **Commit**: `c249378c63232354d17083c83fe94fe728960a27` (HEAD on `main` at vendoring time)
- **Vendored on**: 2026-05-12
- **License**: MIT
- **Reference**: Notwell JH, Wood MW. *ADMET property prediction through combinations of molecular fingerprints.* arXiv:2310.00174 (2023).

MapLight+GNN = MapLight's hand-crafted features (ECFP/Avalon/ErG/RDKit-200) concatenated with a 300-dim GIN supervised-masking embedding from `molfeat`, fed to CatBoost. The GIN encoder is a public DGL-LifeSci pretrained checkpoint; no extra TDC-task training data.

No modifications were made to the upstream files. Our wrapper at `../wrapper/run_all.py` loops the upstream code over all 22 TDC ADMET endpoints reading from the pinned 2026-03-24 snapshot.

"""Re-evaluate MapLight+GNN across all 22 TDC ADMET endpoints.

Reads from the pinned 2026-03-24 TDC snapshot at ../../../data/admet_group/.
Structurally identical to the MapLight wrapper, but imports get_fingerprints
from maplight_gnn which appends GIN supervised-masking embeddings via molfeat.

Output: predictions_<endpoint>.csv per endpoint, plus summary_metrics.csv.
"""
import argparse
import json
import sys
from pathlib import Path

import catboost as cb
import pandas as pd
from tdc.benchmark_group import admet_group
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
SNAPSHOT_DIR = REPO_ROOT / "data"
UPSTREAM_DIR = Path(__file__).resolve().parent.parent / "upstream"
sys.path.insert(0, str(UPSTREAM_DIR))
from maplight_gnn import get_fingerprints, scaler  # noqa: E402

BENCHMARK_CONFIG = {
    "caco2_wang": ("regression", False),
    "bioavailability_ma": ("binary", False),
    "lipophilicity_astrazeneca": ("regression", False),
    "solubility_aqsoldb": ("regression", False),
    "hia_hou": ("binary", False),
    "pgp_broccatelli": ("binary", False),
    "bbb_martins": ("binary", False),
    "ppbr_az": ("regression", False),
    "vdss_lombardo": ("regression", True),
    "cyp2c9_veith": ("binary", False),
    "cyp2d6_veith": ("binary", False),
    "cyp3a4_veith": ("binary", False),
    "cyp2c9_substrate_carbonmangels": ("binary", False),
    "cyp2d6_substrate_carbonmangels": ("binary", False),
    "cyp3a4_substrate_carbonmangels": ("binary", False),
    "half_life_obach": ("regression", True),
    "clearance_hepatocyte_az": ("regression", True),
    "clearance_microsome_az": ("regression", True),
    "ld50_zhu": ("regression", False),
    "herg": ("binary", False),
    "ames": ("binary", False),
    "dili": ("binary", False),
}

SEEDS = [1, 2, 3, 4, 5]


def run_one(benchmark_name: str, group: admet_group, out_dir: Path) -> dict:
    task, log_scale = BENCHMARK_CONFIG[benchmark_name]
    predictions_list = []
    per_seed_preds: list[pd.DataFrame] = []

    for seed in tqdm(SEEDS, desc=benchmark_name, leave=False):
        benchmark = group.get(benchmark_name)
        name = benchmark["name"]
        train, test = benchmark["train_val"], benchmark["test"]

        X_train = get_fingerprints(train["Drug"])
        X_test = get_fingerprints(test["Drug"])

        params = {"random_strength": 2, "random_seed": seed, "verbose": 0}

        if task == "regression":
            y_scaler = scaler(log=log_scale)
            y_scaler.fit(train["Y"].values)
            train["Y_scale"] = y_scaler.transform(train["Y"].values)

            params["loss_function"] = "MAE"
            model = cb.CatBoostRegressor(**params)
            model.fit(X_train, train["Y_scale"].values)
            y_pred = y_scaler.inverse_transform(model.predict(X_test)).reshape(-1)
        else:
            params["loss_function"] = "Logloss"
            model = cb.CatBoostClassifier(**params)
            model.fit(X_train, train["Y"].values)
            y_pred = model.predict_proba(X_test)[:, 1]

        predictions_list.append({name: y_pred})
        per_seed_preds.append(
            pd.DataFrame({"Drug_ID": test["Drug_ID"], "Drug": test["Drug"], "Y_true": test["Y"], f"Y_pred_seed{seed}": y_pred})
        )

    merged = per_seed_preds[0][["Drug_ID", "Drug", "Y_true"]].copy()
    for df in per_seed_preds:
        seed_col = [c for c in df.columns if c.startswith("Y_pred_seed")][0]
        merged[seed_col] = df[seed_col].values
    merged.to_csv(out_dir / f"predictions_{benchmark_name}.csv", index=False)

    results = group.evaluate_many(predictions_list)
    return results[benchmark_name]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", help="single endpoint to run; default = all 22")
    parser.add_argument("--out", default=str(Path(__file__).resolve().parent / "results"))
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    group = admet_group(path=str(SNAPSHOT_DIR))
    endpoints = [args.endpoint] if args.endpoint else list(BENCHMARK_CONFIG.keys())

    summary_rows = []
    for ep in endpoints:
        metric_mean, metric_std = run_one(ep, group, out_dir)
        summary_rows.append({"endpoint": ep, "mean": metric_mean, "std": metric_std})
        print(f"{ep}: {metric_mean:.4f} ± {metric_std:.4f}")

    pd.DataFrame(summary_rows).to_csv(out_dir / "summary_metrics.csv", index=False)
    print(json.dumps(summary_rows, indent=2))


if __name__ == "__main__":
    main()

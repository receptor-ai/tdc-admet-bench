"""Re-evaluate CaliciBoost on the TDC Caco-2 endpoint.

Reads from the pinned 2026-03-24 TDC snapshot at ../../../data/admet_group/.
Swaps the upstream GPU-only XGBoost path (`tree_method='gpu_hist'`) for the
CPU-equivalent `tree_method='hist'` so the run is reproducible without GPU.

Output: predictions_caco2_wang.csv + summary_metrics.csv.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from tdc.benchmark_group import admet_group
from tqdm import tqdm
from xgboost import XGBRegressor

REPO_ROOT = Path(__file__).resolve().parents[3]
SNAPSHOT_DIR = REPO_ROOT / "data"
UPSTREAM_DIR = Path(__file__).resolve().parent.parent / "upstream"
sys.path.insert(0, str(UPSTREAM_DIR))
from calici_boost import (  # noqa: E402
    add_padel_descriptors,
    clean_data,
    featurize,
    xg_parmas,
)

# Upstream's calici_boost.py reads 'PaDEL_Descriptors.csv' via a hard-coded
# relative path. Run with cwd = upstream dir so the read succeeds against the
# vendored copy.
import os  # noqa: E402

os.chdir(UPSTREAM_DIR)

SEEDS = [1, 2, 3, 4, 5]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(Path(__file__).resolve().parent / "results"))
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    group = admet_group(path=str(SNAPSHOT_DIR))
    predictions_list = []
    per_seed_preds: list[pd.DataFrame] = []

    for seed in tqdm(SEEDS, desc="caco2_wang"):
        benchmark = group.get("Caco2_Wang")
        name = benchmark["name"]
        train_val, test = benchmark["train_val"], benchmark["test"]

        train_padel = add_padel_descriptors(train_val, "train_val")
        test_padel = add_padel_descriptors(test, "test")
        train_clean, test_clean = clean_data(train_padel, test_padel)
        x_train, y_train, x_test, _ = featurize(train_clean, test_clean, seed)

        model = XGBRegressor(
            **xg_parmas,
            max_bin=512,
            random_state=seed,
            tree_method="hist",  # was 'gpu_hist' upstream; identical numerically
        )
        model.fit(x_train, y_train, verbose=False)
        y_pred = model.predict(x_test)
        predictions_list.append({name: y_pred})
        per_seed_preds.append(
            pd.DataFrame(
                {
                    "Drug_ID": test["Drug_ID"],
                    "Drug": test["Drug"],
                    "Y_true": test["Y"],
                    f"Y_pred_seed{seed}": y_pred,
                }
            )
        )

    merged = per_seed_preds[0][["Drug_ID", "Drug", "Y_true"]].copy()
    for df in per_seed_preds:
        seed_col = [c for c in df.columns if c.startswith("Y_pred_seed")][0]
        merged[seed_col] = df[seed_col].values
    merged.to_csv(out_dir / "predictions_caco2_wang.csv", index=False)

    results = group.evaluate_many(predictions_list)
    metric_mean, metric_std = results["caco2_wang"]
    pd.DataFrame(
        [{"endpoint": "caco2_wang", "mean": metric_mean, "std": metric_std}]
    ).to_csv(out_dir / "summary_metrics.csv", index=False)
    print(json.dumps({"caco2_wang": [metric_mean, metric_std]}, indent=2))


if __name__ == "__main__":
    main()

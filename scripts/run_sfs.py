#!/usr/bin/env python
"""Run Sequential Feature Selection on a TDC ADMET benchmark."""

import argparse
import json
import logging
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import get_scorer

from tdc_admet_bench.config import (
    suppress_warnings, DEFAULT_FINGERPRINTS,
    BENCHMARK_METRICS, METRIC_TO_SKLEARN, METRIC_TO_TASK,
    get_ranks, compute_selection_score,
)
from tdc_admet_bench.preprocess import preprocess_dataset
from tdc_admet_bench.features import build_feature_matrix
from tdc_admet_bench.models import get_model
from tdc_admet_bench.sfs import SequentialFeatureSelector

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Sequential Feature Selection for TDC ADMET")
    parser.add_argument("--benchmark", default="caco2_wang", help="TDC benchmark name")
    parser.add_argument("--model", default="lgb", help="Model type (lgb, xgb, rf, cat, svm)")
    parser.add_argument("--k-features", type=int, default=10, help="Max feature groups to select")
    parser.add_argument("--forward", action="store_true", default=True, help="Forward selection (default)")
    parser.add_argument("--backward", action="store_true", help="Backward selection")
    parser.add_argument("--floating", action="store_true", help="Enable floating")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON file")
    args = parser.parse_args()

    suppress_warnings()

    benchmark_name = args.benchmark
    model_type = args.model
    forward = not args.backward
    seed = args.seed

    metric = BENCHMARK_METRICS[benchmark_name]
    cv_scoring = METRIC_TO_SKLEARN[metric]
    test_scoring = METRIC_TO_SKLEARN[metric]
    task = METRIC_TO_TASK[metric]
    metric_name = metric.lower()

    logger.info("=" * 80)
    logger.info(f"SFS: {benchmark_name} | {model_type} | {'forward' if forward else 'backward'}")
    logger.info("=" * 80)

    # Load and preprocess
    from tdc_admet_bench.config import load_benchmark
    train_df, test_df = load_benchmark(benchmark_name)
    agg = "mode" if task == "classification" else "mean"
    train_df = preprocess_dataset(train_df, aggregate_duplicates=agg)
    test_df = preprocess_dataset(test_df, aggregate_duplicates=None, filters=[])
    logger.info(f"Train: {len(train_df)}, Test: {len(test_df)}")

    # Build features
    X, feature_groups, feature_names, fixed_group_indices = build_feature_matrix(
        train_df["smiles"].tolist(), fingerprint_configs=DEFAULT_FINGERPRINTS,
    )
    y = train_df["target"].values

    X_test, _, _, _ = build_feature_matrix(
        test_df["smiles"].tolist(), fingerprint_configs=DEFAULT_FINGERPRINTS, verbose=False,
    )
    y_test = test_df["target"].values

    # Model
    model_params = {"n_estimators": 100, "random_state": seed, "n_jobs": -1, "verbose": -1}
    model = get_model(model_type, task=task, params=model_params)

    # CV
    cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=seed)

    # Tracking
    test_scorer = get_scorer(test_scoring)
    best = {"score": float("-inf"), "test": None, "train": None, "train_std": None, "features": None}
    oracle = {"score": None, "test": float("-inf"), "train": None, "train_std": None, "features": None}

    def step_callback(step, result):
        feature_idx = result["feature_idx"]
        cv_scores = result["cv_scores"]
        selected_groups = result["selected_groups"]
        step_type = result.get("step_type", "add")

        train_mean = np.mean(cv_scores)
        train_std = np.std(cv_scores)

        model_step = get_model(model_type, task=task, params=model_params)
        model_step.fit(X[:, feature_idx], y)
        test_score = test_scorer(model_step, X_test[:, feature_idx], y_test)

        selected_names = [feature_names[i] for i in selected_groups]

        if train_mean > best["score"]:
            best.update(score=train_mean, test=test_score, train=train_mean,
                        train_std=train_std, features=selected_names[:])

        if test_score > oracle["test"]:
            oracle.update(score=train_mean, test=test_score, train=train_mean,
                          train_std=train_std, features=selected_names[:])

        logger.info(
            f"Step {step} [{step_type}]: {len(selected_names)} groups | "
            f"CV {metric_name.upper()}: {abs(train_mean):.4f}+-{train_std:.4f} | "
            f"Test {metric_name.upper()}: {abs(test_score):.4f}"
        )

    # Run SFS
    sfs = SequentialFeatureSelector(
        model, k_features=args.k_features, cv=cv, scoring=cv_scoring,
        feature_groups=feature_groups, fixed_group_indices=fixed_group_indices,
        callback=step_callback, forward=forward, floating=args.floating,
        max_features=X.shape[1], feature_names=feature_names,
    )
    sfs.fit(X, y)

    # Results
    best_rank = get_ranks({benchmark_name: [abs(best["test"]), 0]})[benchmark_name]
    oracle_rank = get_ranks({benchmark_name: [abs(oracle["test"]), 0]})[benchmark_name]

    print("\n" + "=" * 80)
    print("SFS Complete!")
    print("=" * 80)
    print(f"Best (by CV score):")
    print(f"  CV {metric_name.upper()}: {abs(best['train']):.4f}+-{best['train_std']:.4f}")
    print(f"  Test {metric_name.upper()}: {abs(best['test']):.4f} | Rank: {best_rank['rank']}/{best_rank['total_models']}")
    print(f"  Features: {best['features']}")
    print(f"Oracle (by test score):")
    print(f"  CV {metric_name.upper()}: {abs(oracle['train']):.4f}+-{oracle['train_std']:.4f}")
    print(f"  Test {metric_name.upper()}: {abs(oracle['test']):.4f} | Rank: {oracle_rank['rank']}/{oracle_rank['total_models']}")
    print(f"  Features: {oracle['features']}")

    if args.output:
        result = {
            "benchmark": benchmark_name,
            "model_type": model_type,
            "best_features": best["features"],
            "best_test": abs(best["test"]),
            "best_cv_mean": abs(best["train"]),
            "best_rank": best_rank["rank"],
            "oracle_features": oracle["features"],
            "oracle_test": abs(oracle["test"]),
            "oracle_rank": oracle_rank["rank"],
        }
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()

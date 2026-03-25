"""Multi-seed TDC benchmark evaluation."""

import logging

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tdc.benchmark_group import admet_group

from tdc_admet_bench.config import (
    DATA_DIR, BENCHMARK_CONFIG, BENCHMARK_METRICS,
    METRIC_TO_TASK, get_ranks,
)
from tdc_admet_bench.preprocess import preprocess_dataset
from tdc_admet_bench.features import build_feature_matrix
from tdc_admet_bench.models import get_model

logger = logging.getLogger(__name__)

TDC_SEEDS = [1, 2, 3, 4, 5]


def evaluate_benchmark(
    benchmark_name,
    model_type="lgb",
    model_params=None,
    fingerprint_configs=None,
    rdkit_descriptors=None,
    mordred_descriptors=None,
    n_seeds=5,
):
    """Evaluate on a single TDC benchmark with multiple seeds.

    Uses TDC's official evaluation protocol:
    - Train with n different random seeds
    - Predict on official test set
    - TDC evaluate_many() computes official metrics

    Returns:
        Dict from TDC, e.g. {'caco2_wang': [0.331, 0.015]}
    """
    if model_params is None:
        model_params = {"n_estimators": 100, "n_jobs": -1, "verbose": -1}

    metric = BENCHMARK_METRICS[benchmark_name]
    task_type, _ = BENCHMARK_CONFIG[benchmark_name]
    task = METRIC_TO_TASK[metric]
    is_classification = task == "classification"

    logger.info(f"Evaluating {benchmark_name} ({task}, {metric}) with {n_seeds} seeds")

    group = admet_group(path=str(DATA_DIR))
    benchmark = group.get(benchmark_name)

    predictions_list = []
    for seed in TDC_SEEDS[:n_seeds]:
        logger.info(f"  Seed {seed}/{n_seeds}")

        # Preprocess train
        train_df = benchmark["train_val"].rename(columns={"Drug_ID": "id", "Drug": "smiles", "Y": "target"})
        agg = "mode" if is_classification else "mean"
        train_df = preprocess_dataset(train_df, aggregate_duplicates=agg)

        # Preprocess test
        test_df = benchmark["test"].rename(columns={"Drug_ID": "id", "Drug": "smiles", "Y": "target"})
        test_df = preprocess_dataset(test_df, aggregate_duplicates=None, filters=[])

        # Build features
        X_train, _, _, _ = build_feature_matrix(
            train_df["smiles"].tolist(),
            fingerprint_configs=fingerprint_configs,
            rdkit_descriptors=rdkit_descriptors,
            mordred_descriptors=mordred_descriptors,
            verbose=seed == 1,
        )
        y_train = train_df["target"].values

        X_test, _, _, _ = build_feature_matrix(
            test_df["smiles"].tolist(),
            fingerprint_configs=fingerprint_configs,
            rdkit_descriptors=rdkit_descriptors,
            mordred_descriptors=mordred_descriptors,
            verbose=False,
        )

        # Train model
        seed_params = {**model_params, "random_state": seed}
        model = get_model(model_type, task=task, params=seed_params)
        model.fit(X_train, y_train)

        # Predict
        if is_classification:
            if hasattr(model, "predict_proba"):
                y_pred = model.predict_proba(X_test)[:, 1]
            else:
                y_pred = model.decision_function(X_test)
        else:
            y_pred = model.predict(X_test)

        predictions_list.append({benchmark["name"]: y_pred})

    results = group.evaluate_many(predictions_list)
    logger.info(f"  Results: {results}")
    return results


def evaluate_all_benchmarks(
    model_type="lgb",
    model_params=None,
    fingerprint_configs=None,
    rdkit_descriptors=None,
    mordred_descriptors=None,
    n_seeds=5,
):
    """Evaluate on all 22 ADMET benchmarks."""
    all_results = {}
    benchmarks = list(BENCHMARK_CONFIG.keys())

    for i, name in enumerate(benchmarks, 1):
        logger.info(f"\n[{i}/{len(benchmarks)}] {name}")
        results = evaluate_benchmark(
            name,
            model_type=model_type,
            model_params=model_params,
            fingerprint_configs=fingerprint_configs,
            rdkit_descriptors=rdkit_descriptors,
            mordred_descriptors=mordred_descriptors,
            n_seeds=n_seeds,
        )
        all_results.update(results)

    return all_results

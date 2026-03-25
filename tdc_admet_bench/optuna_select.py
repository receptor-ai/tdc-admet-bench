"""Optuna-based feature selection with Bayesian optimisation."""

import logging

import numpy as np
from typing import List, Dict, Optional

import optuna
from sklearn.model_selection import cross_val_score, RepeatedKFold
from sklearn.metrics import get_scorer

from tdc_admet_bench.config import (
    BENCHMARK_METRICS, METRIC_TO_SKLEARN, METRIC_TO_TASK,
    get_ranks, compute_selection_score, load_benchmark,
)
from tdc_admet_bench.preprocess import preprocess_dataset
from tdc_admet_bench.models import get_model
from tdc_admet_bench.features import build_feature_matrix

logger = logging.getLogger(__name__)


def create_feature_objective(
    X, y, X_test, y_test,
    feature_groups, feature_names, fixed_group_indices,
    model_type, model_params, task,
    cv_scoring, test_scoring, cv,
    max_features, selection_weights,
    study,
):
    """Create Optuna objective for feature group selection."""
    test_scorer = get_scorer(test_scoring)
    fixed_set = set(fixed_group_indices)
    n_groups = len(feature_groups)

    def objective(trial):
        selected_groups = list(fixed_group_indices)

        for i in range(n_groups):
            if i in fixed_set:
                continue
            if trial.suggest_categorical(f"use_{feature_names[i]}", [True, False]):
                selected_groups.append(i)

        # Must have at least one group beyond fixed
        if len(selected_groups) == len(fixed_group_indices):
            candidates = [i for i in range(n_groups) if i not in fixed_set]
            if candidates:
                selected_groups.append(candidates[0])

        # Skip duplicates
        selected_names = [feature_names[i] for i in selected_groups]
        params_key = tuple(sorted(selected_names))
        for t in study.trials:
            if t.state == optuna.trial.TrialState.COMPLETE:
                past_names = t.user_attrs.get("selected_names", [])
                if tuple(sorted(past_names)) == params_key:
                    raise optuna.TrialPruned()

        feature_idx = []
        for gi in selected_groups:
            feature_idx.extend(feature_groups[gi])

        X_sel = X[:, feature_idx]
        X_test_sel = X_test[:, feature_idx]

        model = get_model(model_type, task=task, params=model_params)
        cv_scores = cross_val_score(model, X_sel, y, cv=cv, scoring=cv_scoring, n_jobs=1)
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)

        trial.set_user_attr("selected_groups", selected_groups)
        trial.set_user_attr("selected_names", selected_names)
        trial.set_user_attr("n_features", len(feature_idx))
        trial.set_user_attr("cv_mean", cv_mean)
        trial.set_user_attr("cv_std", cv_std)

        model_test = get_model(model_type, task=task, params=model_params)
        model_test.fit(X_sel, y)
        test_score = test_scorer(model_test, X_test_sel, y_test)
        trial.set_user_attr("test_score", test_score)

        w = selection_weights
        return compute_selection_score(
            cv_mean, cv_std, test_score,
            len(feature_idx), max_features,
            train_mean_weight=w.get("train_mean", 1.0),
            train_std_weight=w.get("train_std", 0.0),
            test_score_weight=w.get("test_score", 0.0),
            feature_penalty=w.get("feature_penalty", 0.0),
        )

    return objective


def run_optuna_feature_selection(
    benchmark_name: str,
    model_type: str = "lgb",
    model_params: Optional[dict] = None,
    fingerprint_configs: Optional[List[dict]] = None,
    rdkit_descriptors: Optional[List[str]] = None,
    mordred_descriptors: Optional[List[str]] = None,
    fixed_feature_names: Optional[List[str]] = None,
    n_trials: int = 100,
    timeout: Optional[int] = None,
    seed: int = 42,
    selection_weights: Optional[dict] = None,
) -> Dict:
    """Run Optuna-based feature selection on a single benchmark.

    Args:
        benchmark_name: TDC benchmark name.
        model_type: Model type string (e.g. 'lgb').
        model_params: Model hyperparameters.
        fingerprint_configs: List of fingerprint config dicts.
        rdkit_descriptors: RDKit descriptor names for individual selection.
        mordred_descriptors: Mordred descriptor names for individual selection.
        fixed_feature_names: Features always included.
        n_trials: Number of Optuna trials.
        timeout: Timeout in seconds.
        seed: Random seed.
        selection_weights: Dict with train_mean, train_std, test_score, feature_penalty.

    Returns:
        Dict with best results and ranking.
    """
    if model_params is None:
        model_params = {"n_estimators": 100, "random_state": seed, "n_jobs": -1, "verbose": -1}
    if fingerprint_configs is None:
        from tdc_admet_bench.config import DEFAULT_FINGERPRINTS
        fingerprint_configs = DEFAULT_FINGERPRINTS
    if fixed_feature_names is None:
        fixed_feature_names = []
    if selection_weights is None:
        selection_weights = {"train_mean": 1.0, "train_std": 0.0, "test_score": 0.0, "feature_penalty": 0.0}

    metric = BENCHMARK_METRICS[benchmark_name]
    cv_scoring = METRIC_TO_SKLEARN[metric]
    test_scoring = METRIC_TO_SKLEARN[metric]
    task = METRIC_TO_TASK[metric]
    metric_name = metric.lower()

    logger.info("=" * 80)
    logger.info(f"Optuna Feature Selection: {benchmark_name}")
    logger.info(f"Model: {model_type} | Task: {task} | Trials: {n_trials}")
    logger.info("=" * 80)

    # Load and preprocess
    train_df, test_df = load_benchmark(benchmark_name)
    agg = "mode" if task == "classification" else "mean"
    train_df = preprocess_dataset(train_df, aggregate_duplicates=agg)
    test_df = preprocess_dataset(test_df, aggregate_duplicates=None, filters=[])
    logger.info(f"Train: {len(train_df)}, Test: {len(test_df)}")

    # Build features
    X, feature_groups, feature_names, fixed_group_indices = build_feature_matrix(
        train_df["smiles"].tolist(),
        fingerprint_configs=fingerprint_configs,
        rdkit_descriptors=rdkit_descriptors,
        mordred_descriptors=mordred_descriptors,
        fixed_feature_names=fixed_feature_names,
    )
    y = train_df["target"].values

    X_test, _, _, _ = build_feature_matrix(
        test_df["smiles"].tolist(),
        fingerprint_configs=fingerprint_configs,
        rdkit_descriptors=rdkit_descriptors,
        mordred_descriptors=mordred_descriptors,
        fixed_feature_names=fixed_feature_names,
        verbose=False,
    )
    y_test = test_df["target"].values

    # CV
    cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=seed)

    # Optuna study
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )

    objective = create_feature_objective(
        X, y, X_test, y_test,
        feature_groups, feature_names, fixed_group_indices,
        model_type, model_params, task,
        cv_scoring, test_scoring, cv,
        X.shape[1], selection_weights, study,
    )

    # Tracking
    best = {"score": float("-inf"), "test": None, "train": None, "train_std": None, "features": None, "trial": None}
    oracle = {"score": None, "test": float("-inf"), "train": None, "train_std": None, "features": None, "trial": None}

    def trial_callback(study, trial):
        nonlocal best, oracle
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return
        sel_score = trial.value
        train_mean = trial.user_attrs["cv_mean"]
        train_std = trial.user_attrs["cv_std"]
        test_score = trial.user_attrs["test_score"]
        names = trial.user_attrs["selected_names"]

        if sel_score > best["score"]:
            best.update(score=sel_score, test=test_score, train=train_mean, train_std=train_std,
                        features=names[:], trial=trial.number)
        if test_score > oracle["test"]:
            oracle.update(score=sel_score, test=test_score, train=train_mean, train_std=train_std,
                          features=names[:], trial=trial.number)

        logger.info(
            f"Trial {trial.number}: {len(names)} groups | "
            f"CV {metric_name.upper()}: {abs(train_mean):.4f}+-{train_std:.4f} | "
            f"Test {metric_name.upper()}: {abs(test_score):.4f}"
        )

    study.optimize(objective, n_trials=n_trials, timeout=timeout, callbacks=[trial_callback])

    # Ranks
    best_rank = get_ranks({benchmark_name: [abs(best["test"]), 0]})[benchmark_name]
    oracle_rank = get_ranks({benchmark_name: [abs(oracle["test"]), 0]})[benchmark_name]

    logger.info("=" * 80)
    logger.info("Optuna Complete!")
    logger.info("=" * 80)
    logger.info(
        f"Best (trial {best['trial']}): "
        f"CV {metric_name.upper()}: {abs(best['train']):.4f}+-{best['train_std']:.4f} | "
        f"Test {metric_name.upper()}: {abs(best['test']):.4f} | "
        f"Rank: {best_rank['rank']}/{best_rank['total_models']}"
    )
    logger.info(f"  Features: {best['features']}")
    logger.info(
        f"Oracle (trial {oracle['trial']}): "
        f"CV {metric_name.upper()}: {abs(oracle['train']):.4f}+-{oracle['train_std']:.4f} | "
        f"Test {metric_name.upper()}: {abs(oracle['test']):.4f} | "
        f"Rank: {oracle_rank['rank']}/{oracle_rank['total_models']}"
    )
    logger.info(f"  Features: {oracle['features']}")

    return {
        "benchmark": benchmark_name,
        "model_type": model_type,
        "metric_name": metric_name,
        "selected_features": best["features"],
        "test_score": abs(best["test"]),
        "cv_mean": abs(best["train"]),
        "cv_std": best["train_std"],
        "rank": best_rank["rank"],
        "total_models": best_rank["total_models"],
        "n_trials": n_trials,
    }

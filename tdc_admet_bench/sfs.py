"""Sequential Feature Selection with forward/backward and floating support."""

import logging

import numpy as np
from typing import List, Callable, Optional
from sklearn.model_selection import cross_val_score, RepeatedKFold
from sklearn.metrics import get_scorer
from sklearn.base import clone
from tqdm import tqdm

from tdc_admet_bench.config import compute_selection_score

logger = logging.getLogger(__name__)


class SequentialFeatureSelector:
    """Sequential feature selection with feature groups.

    Modes:
    - forward=True,  floating=False: Forward selection
    - forward=True,  floating=True:  SFFS (forward + try removing)
    - forward=False, floating=False: Backward selection
    - forward=False, floating=True:  SBFS (backward + try adding)

    Parameters
    ----------
    estimator : sklearn-compatible model
    k_features : int
        Target number of feature groups to select (excl. fixed).
    cv : CV splitter or int
        Cross-validation strategy.  Defaults to 5x5 RepeatedKFold.
    scoring : str
        sklearn scoring string.
    feature_groups : list of list of int
        Each sub-list groups column indices that belong together.
    fixed_group_indices : list of int
        Group indices always included (not counted in k_features).
    callback : callable
        Called after each step: callback(step, result_dict).
    forward : bool
        If True, add features; if False, remove features.
    floating : bool
        If True, try reverse operation after each main step.
    selection_weights : dict
        Keys: train_mean, train_std, test_score, feature_penalty.
    max_features : int
        Total feature count for penalty calculation.
    X_test, y_test : arrays
        Test data (only used if test_score weight != 0).
    feature_names : list of str
        Names for each group (for logging).
    """

    def __init__(
        self,
        estimator,
        k_features: int,
        cv=None,
        scoring: str = "neg_mean_absolute_error",
        feature_groups: Optional[List[List[int]]] = None,
        fixed_group_indices: Optional[List[int]] = None,
        callback: Optional[Callable] = None,
        forward: bool = True,
        floating: bool = False,
        selection_weights: Optional[dict] = None,
        max_features: Optional[int] = None,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ):
        self.estimator = estimator
        self.k_features = k_features
        self.cv = cv if cv is not None else RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)
        self.scoring = scoring
        self.feature_groups = feature_groups
        self.fixed_group_indices = fixed_group_indices or []
        self.callback = callback
        self.forward = forward
        self.floating = floating
        self.selection_weights = selection_weights or {
            "train_mean": 1.0, "train_std": 0.0,
            "test_score": 0.0, "feature_penalty": 0.0,
        }
        self.max_features = max_features
        self.X_test = X_test
        self.y_test = y_test
        self.feature_names = feature_names or []

        self.scorer_ = get_scorer(scoring) if scoring else None

        # Results (set during fit)
        self.selected_groups_ = []
        self.selected_features_ = []
        self.subsets_ = {}
        self.step_ = 0
        self.best_score_by_size_ = {}

    def _get_features_from_groups(self, group_indices):
        features = []
        for idx in group_indices:
            features.extend(self.feature_groups[idx])
        return features

    def _compute_score(self, X, y, feature_indices):
        cv_scores = cross_val_score(
            clone(self.estimator), X[:, feature_indices], y,
            cv=self.cv, scoring=self.scoring, n_jobs=1,
        )
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)

        test_score = 0.0
        w = self.selection_weights
        if w.get("test_score", 0.0) != 0.0 and self.X_test is not None:
            model = clone(self.estimator)
            model.fit(X[:, feature_indices], y)
            test_score = self.scorer_(model, self.X_test[:, feature_indices], self.y_test)

        score = compute_selection_score(
            cv_mean, cv_std, test_score,
            len(feature_indices), self.max_features,
            train_mean_weight=w.get("train_mean", 1.0),
            train_std_weight=w.get("train_std", 0.0),
            test_score_weight=w.get("test_score", 0.0),
            feature_penalty=w.get("feature_penalty", 0.0),
        )
        return score, cv_mean, cv_scores

    def fit(self, X, y):
        if self.forward:
            self._fit_forward(X, y)
        else:
            self._fit_backward(X, y)
        self.selected_features_ = self._get_features_from_groups(self.selected_groups_)
        return self

    # ----- forward -----
    def _fit_forward(self, X, y):
        if self.fixed_group_indices:
            self.selected_groups_ = list(self.fixed_group_indices)
            fixed_names = [self.feature_names[i] for i in self.fixed_group_indices] if self.feature_names else self.fixed_group_indices
            logger.info(f"Forward SFS: fixed features: {fixed_names}")
        else:
            self.selected_groups_ = []

        current_score = None
        excluded = set()
        n_fixed = len(self.fixed_group_indices)

        try:
            while len(self.selected_groups_) - n_fixed < self.k_features:
                step = len(self.selected_groups_) - n_fixed + 1
                candidates = [
                    i for i in range(len(self.feature_groups))
                    if i not in self.selected_groups_ and i not in excluded
                ]
                if not candidates:
                    break

                best_score = -np.inf
                best_idx = None
                best_cv_scores = None

                pbar = tqdm(candidates, desc=f"Step {step} [ADD]", unit="group", leave=False)
                for cand_idx in pbar:
                    test_groups = self.selected_groups_ + [cand_idx]
                    test_features = self._get_features_from_groups(test_groups)
                    score, cv_mean, cv_scores = self._compute_score(X, y, test_features)
                    if score > best_score:
                        best_score = score
                        best_idx = cand_idx
                        best_cv_scores = cv_scores
                        pbar.set_postfix({"best_cv": f"{abs(cv_mean):.4f}"})
                pbar.close()

                self.selected_groups_.append(best_idx)
                current_score = best_score
                self._record_step(X, y, "add", cv_scores=best_cv_scores)

                if self.floating:
                    removed = self._floating_remove(X, y, current_score, just_added=best_idx)
                    excluded.update(removed)

        except KeyboardInterrupt:
            logger.info("\nInterrupted! Returning current results...")

    # ----- backward -----
    def _fit_backward(self, X, y):
        self.selected_groups_ = list(range(len(self.feature_groups)))
        logger.info(f"Backward SFS: starting with all {len(self.selected_groups_)} groups")

        current_features = self._get_features_from_groups(self.selected_groups_)
        current_score, _, _ = self._compute_score(X, y, current_features)
        n_fixed = len(self.fixed_group_indices)
        target_size = self.k_features + n_fixed

        try:
            while len(self.selected_groups_) > target_size:
                step = len(self.feature_groups) - len(self.selected_groups_) + 1
                removable = [i for i in self.selected_groups_ if i not in self.fixed_group_indices]
                if not removable:
                    break

                best_score = -np.inf
                best_idx = None
                best_cv_scores = None

                pbar = tqdm(removable, desc=f"Step {step} [REMOVE]", unit="group", leave=False)
                for rem_idx in pbar:
                    test_groups = [g for g in self.selected_groups_ if g != rem_idx]
                    test_features = self._get_features_from_groups(test_groups)
                    score, cv_mean, cv_scores = self._compute_score(X, y, test_features)
                    if score > best_score:
                        best_score = score
                        best_idx = rem_idx
                        best_cv_scores = cv_scores
                        pbar.set_postfix({"best_cv": f"{abs(cv_mean):.4f}"})
                pbar.close()

                self.selected_groups_.remove(best_idx)
                current_score = best_score
                self._record_step(X, y, "remove", cv_scores=best_cv_scores)

                if self.floating:
                    self._floating_add(X, y, current_score, just_removed=best_idx)

        except KeyboardInterrupt:
            logger.info("\nInterrupted! Returning current results...")

    # ----- floating helpers -----
    def _floating_remove(self, X, y, current_score, just_added):
        removed = set()
        while True:
            removable = [
                i for i in self.selected_groups_
                if i not in self.fixed_group_indices and i != just_added
            ]
            if len(removable) <= 1:
                break

            best_score = current_score
            best_idx = None
            best_cv_scores = None

            for rem_idx in removable:
                test_groups = [g for g in self.selected_groups_ if g != rem_idx]
                test_features = self._get_features_from_groups(test_groups)
                score, _, cv_scores = self._compute_score(X, y, test_features)
                if score > best_score:
                    best_score = score
                    best_idx = rem_idx
                    best_cv_scores = cv_scores

            if best_idx is not None:
                new_size = len(self.selected_groups_) - 1
                if new_size in self.best_score_by_size_ and best_score <= self.best_score_by_size_[new_size]:
                    break
                self.selected_groups_.remove(best_idx)
                removed.add(best_idx)
                current_score = best_score
                name = self.feature_names[best_idx] if self.feature_names else best_idx
                logger.info(f"  Float: removed '{name}', score {best_score:.4f}")
                self._record_step(X, y, "float_remove", cv_scores=best_cv_scores)
            else:
                break
        return removed

    def _floating_add(self, X, y, current_score, just_removed):
        while True:
            candidates = [
                i for i in range(len(self.feature_groups))
                if i not in self.selected_groups_ and i != just_removed
            ]
            if not candidates:
                break

            best_score = current_score
            best_idx = None
            best_cv_scores = None

            for cand_idx in candidates:
                test_groups = self.selected_groups_ + [cand_idx]
                test_features = self._get_features_from_groups(test_groups)
                score, _, cv_scores = self._compute_score(X, y, test_features)
                if score > best_score:
                    best_score = score
                    best_idx = cand_idx
                    best_cv_scores = cv_scores

            if best_idx is not None:
                new_size = len(self.selected_groups_) + 1
                if new_size in self.best_score_by_size_ and best_score <= self.best_score_by_size_[new_size]:
                    break
                self.selected_groups_.append(best_idx)
                current_score = best_score
                name = self.feature_names[best_idx] if self.feature_names else best_idx
                logger.info(f"  Float: added '{name}', score {best_score:.4f}")
                self._record_step(X, y, "float_add", cv_scores=best_cv_scores)
            else:
                break

    # ----- recording -----
    def _record_step(self, X, y, step_type, cv_scores=None):
        self.step_ += 1
        current_features = self._get_features_from_groups(self.selected_groups_)

        if cv_scores is not None:
            avg_score = np.mean(cv_scores)
        else:
            X_subset = X[:, current_features]
            cv_scores = cross_val_score(
                self.estimator, X_subset, y,
                cv=self.cv, scoring=self.scoring, n_jobs=1,
            )
            avg_score = np.mean(cv_scores)

        self.subsets_[self.step_] = {
            "feature_idx": tuple(current_features),
            "cv_scores": cv_scores,
            "avg_score": avg_score,
            "step_type": step_type,
        }

        n_groups = len(self.selected_groups_)
        if n_groups not in self.best_score_by_size_ or avg_score > self.best_score_by_size_[n_groups]:
            self.best_score_by_size_[n_groups] = avg_score

        if self.callback:
            self.callback(self.step_, {
                "feature_idx": tuple(current_features),
                "selected_groups": list(self.selected_groups_),
                "cv_scores": cv_scores,
                "avg_score": avg_score,
                "step_type": step_type,
            })

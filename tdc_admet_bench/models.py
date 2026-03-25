"""Model registry for ML models."""

from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    HistGradientBoostingRegressor, HistGradientBoostingClassifier,
)
from sklearn.svm import SVR, SVC
from sklearn.linear_model import Ridge, LogisticRegression
from lightgbm import LGBMRegressor, LGBMClassifier
from xgboost import XGBRegressor, XGBClassifier
from catboost import CatBoostRegressor, CatBoostClassifier

REGRESSION_MODELS = {
    "rf": RandomForestRegressor,
    "svm": SVR,
    "ridge": Ridge,
    "lgb": LGBMRegressor,
    "xgb": XGBRegressor,
    "cat": CatBoostRegressor,
    "hist": HistGradientBoostingRegressor,
}

CLASSIFICATION_MODELS = {
    "rf": RandomForestClassifier,
    "svm": SVC,
    "logistic": LogisticRegression,
    "lgb": LGBMClassifier,
    "xgb": XGBClassifier,
    "cat": CatBoostClassifier,
    "hist": HistGradientBoostingClassifier,
}


def get_model(model_type, task="regression", params=None):
    """Instantiate a model from the registry.

    Args:
        model_type: One of 'rf', 'lgb', 'xgb', 'cat', 'svm', 'ridge', 'hist'.
        task: 'regression' or 'classification'.
        params: Dict of model hyperparameters.
    """
    if params is None:
        params = {}

    registry = REGRESSION_MODELS if task == "regression" else CLASSIFICATION_MODELS

    if model_type not in registry:
        raise ValueError(
            f"Unknown model '{model_type}' for task '{task}'. "
            f"Available: {sorted(registry.keys())}"
        )

    return registry[model_type](**params)

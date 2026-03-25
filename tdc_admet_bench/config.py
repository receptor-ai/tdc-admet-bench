"""Project configuration, benchmark metadata, and utility functions."""

import bisect
import json
import logging
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import make_scorer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)


def suppress_warnings():
    """Suppress common ML library warnings (RDKit, sklearn, LightGBM)."""
    from rdkit import RDLogger
    RDLogger.logger().setLevel(RDLogger.ERROR)
    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.validation")

# ---------------------------------------------------------------------------
# Benchmark metadata
# ---------------------------------------------------------------------------
# (task_type, log_transform)
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

# Benchmark name -> TDC metric name
BENCHMARK_METRICS = {
    "caco2_wang": "MAE",
    "bioavailability_ma": "AUROC",
    "lipophilicity_astrazeneca": "MAE",
    "solubility_aqsoldb": "MAE",
    "hia_hou": "AUROC",
    "pgp_broccatelli": "AUROC",
    "bbb_martins": "AUROC",
    "ppbr_az": "MAE",
    "vdss_lombardo": "SPEARMAN",
    "cyp2c9_veith": "AUPRC",
    "cyp2d6_veith": "AUPRC",
    "cyp3a4_veith": "AUPRC",
    "cyp2c9_substrate_carbonmangels": "AUPRC",
    "cyp2d6_substrate_carbonmangels": "AUROC",
    "cyp3a4_substrate_carbonmangels": "AUPRC",
    "half_life_obach": "SPEARMAN",
    "clearance_hepatocyte_az": "SPEARMAN",
    "clearance_microsome_az": "SPEARMAN",
    "ld50_zhu": "MAE",
    "herg": "AUROC",
    "ames": "AUROC",
    "dili": "AUROC",
}

# ---------------------------------------------------------------------------
# Metric mappings
# ---------------------------------------------------------------------------
def _spearman_corr(y_true, y_pred):
    return spearmanr(y_true, y_pred)[0]

SPEARMAN_SCORER = make_scorer(_spearman_corr, greater_is_better=True)

METRIC_TO_SKLEARN = {
    "MAE": "neg_mean_absolute_error",
    "SPEARMAN": SPEARMAN_SCORER,
    "AUROC": "roc_auc",
    "AUPRC": "average_precision",
}

METRIC_TO_TASK = {
    "MAE": "regression",
    "SPEARMAN": "regression",
    "AUROC": "classification",
    "AUPRC": "classification",
}

# ---------------------------------------------------------------------------
# Default fingerprint configs for feature selection
# ---------------------------------------------------------------------------
DEFAULT_FINGERPRINTS = [
    # Configurable (binary, with length)
    {"kind": "ecfp", "radius": 2, "length": 1024},
    {"kind": "fcfp", "radius": 2, "length": 1024},
    {"kind": "avalon", "length": 1024},
    {"kind": "rdkit", "length": 1024},
    {"kind": "topological", "length": 1024},
    {"kind": "atompair", "length": 1024},
    {"kind": "pattern", "length": 1024},
    {"kind": "layered", "length": 1024},
    {"kind": "secfp", "length": 1024},
    # Counting fingerprints
    {"kind": "ecfp-count", "radius": 2, "length": 1024},
    {"kind": "fcfp-count", "radius": 2, "length": 1024},
    {"kind": "rdkit-count", "length": 1024},
    {"kind": "topological-count", "length": 1024},
    {"kind": "atompair-count", "length": 1024},
    # Fixed-size (2D only)
    {"kind": "maccs"},       # 167
    {"kind": "erg"},         # 315
    {"kind": "estate"},      # 79
    {"kind": "desc2D"},      # 223
    {"kind": "cats2D"},      # 189
    {"kind": "scaffoldkeys"},  # 42
    {"kind": "skeys"},       # 42
]

# ---------------------------------------------------------------------------
# Leaderboard ranking
# ---------------------------------------------------------------------------
def get_ranks(results):
    """Get leaderboard ranks for TDC results.

    Args:
        results: Dict from TDC evaluate_many, e.g. {'caco2_wang': [0.331, 0.0]}

    Returns:
        Dict mapping benchmark names to rank info dicts.
    """
    leaderboard_path = DATA_DIR / "tdc_admet_leaderboard.json"
    leaderboard_data = json.loads(leaderboard_path.read_text())

    ranks = {}
    for benchmark_name, metric_values in results.items():
        if benchmark_name not in leaderboard_data:
            ranks[benchmark_name] = None
            continue

        benchmark = leaderboard_data[benchmark_name]
        entries = benchmark["entries"]
        metric_name = benchmark["metric_name"]
        metric_mean = metric_values[0]
        metric_std = metric_values[1] if len(metric_values) > 1 else None

        leaderboard_scores = [entry["metric_mean"] for entry in entries]
        lower_is_better = metric_name.upper() in ("MAE", "MSE", "RMSE", "MAD")

        if lower_is_better:
            rank = bisect.bisect_right(leaderboard_scores, metric_mean) + 1
        else:
            rank = sum(1 for s in leaderboard_scores if s > metric_mean) + 1

        total_models = len(entries)
        ranks[benchmark_name] = {
            "rank": rank,
            "total_models": total_models,
            "metric_name": metric_name,
            "metric_mean": metric_mean,
            "metric_std": metric_std,
        }
    return ranks

# ---------------------------------------------------------------------------
# Selection scoring
# ---------------------------------------------------------------------------
def compute_selection_score(train_mean, train_std, test_score, n_features, max_features,
                            train_mean_weight=1.0, train_std_weight=0.0,
                            test_score_weight=0.0, feature_penalty=0.0):
    """Compute selection score for picking best SFS step or Optuna trial.

    For positive base (e.g., AUROC): multiply by penalty (reduces score).
    For negative base (e.g., neg_MAE): divide by penalty (makes more negative).
    """
    base = (train_mean_weight * train_mean
            + train_std_weight * train_std
            + test_score_weight * test_score)
    multiplier = np.exp(-feature_penalty * n_features / max_features)
    if base >= 0:
        return base * multiplier
    else:
        return base / multiplier

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_benchmark(benchmark_name):
    """Load TDC ADMET benchmark with standardised column names.

    Returns:
        (train_df, test_df) with columns ['id', 'smiles', 'target']
    """
    from tdc.benchmark_group import admet_group
    group = admet_group(path=str(DATA_DIR))
    benchmark = group.get(benchmark_name)
    mapping = {"Drug_ID": "id", "Drug": "smiles", "Y": "target"}
    train_df = benchmark["train_val"].rename(columns=mapping)
    test_df = benchmark["test"].rename(columns=mapping)
    return train_df, test_df

"""Molecular feature transformers and feature matrix builder."""

import logging
import time
from typing import List, Optional

import numpy as np
import pandas as pd
from molfeat.trans import FPVecTransformer, MoleculeTransformer
from molfeat.calc import RDKitDescriptors2D, MordredDescriptors
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Transformers
# ---------------------------------------------------------------------------
class FingerprintTransformer:
    """Fingerprint transformer wrapping molfeat FPVecTransformer.

    Parameters
    ----------
    kind : str
        Fingerprint type (e.g. 'ecfp', 'maccs', 'avalon').
    **params
        Fingerprint-specific parameters (e.g. radius=2, length=2048).
    """

    def __init__(self, kind: str, **params):
        self.kind = kind
        self.params = params

    def fit(self, X, y=None):
        return self

    def transform(self, X: List[str]) -> np.ndarray:
        transformer = FPVecTransformer(kind=self.kind, **self.params)
        return transformer.transform(X)

    def fit_transform(self, X: List[str], y=None) -> np.ndarray:
        return self.transform(X)


class RDKitDescriptorTransformer:
    """RDKit 2D descriptors. Computes all 223, then filters to requested names."""

    def __init__(self, descriptors: List[str]):
        self.descriptors = descriptors

    def fit(self, X, y=None):
        return self

    def transform(self, X: List[str]) -> np.ndarray:
        all_df = _compute_rdkit_descriptors(X)
        return all_df[self.descriptors].values

    def fit_transform(self, X: List[str], y=None) -> np.ndarray:
        return self.transform(X)


class MordredDescriptorTransformer:
    """Mordred 2D descriptors. Computes all ~1613, then filters to requested names."""

    def __init__(self, descriptors: List[str], ignore_3d: bool = True):
        self.descriptors = descriptors
        self.ignore_3d = ignore_3d

    def fit(self, X, y=None):
        return self

    def transform(self, X: List[str]) -> np.ndarray:
        all_df = _compute_mordred_descriptors(X, self.ignore_3d)
        return all_df[self.descriptors].values

    def fit_transform(self, X: List[str], y=None) -> np.ndarray:
        return self.transform(X)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _suppress_rdkit():
    from rdkit import RDLogger
    RDLogger.logger().setLevel(RDLogger.ERROR)


def _compute_rdkit_descriptors(smiles: List[str]) -> pd.DataFrame:
    calc = RDKitDescriptors2D(descrs=None)
    transformer = MoleculeTransformer(
        featurizer=calc, n_jobs=-1, dtype=np.float32,
        parallel_kwargs={"initializer": _suppress_rdkit},
    )
    result = transformer.transform(smiles).astype(np.float32)
    return pd.DataFrame(result, columns=calc.columns)


def _compute_mordred_descriptors(smiles: List[str], ignore_3d: bool = True) -> pd.DataFrame:
    calc = MordredDescriptors(ignore_3D=ignore_3d)
    transformer = MoleculeTransformer(
        featurizer=calc, n_jobs=-1, dtype=np.float32,
        parallel_kwargs={"initializer": _suppress_rdkit},
    )
    result = transformer.transform(smiles).astype(np.float32)
    return pd.DataFrame(result, columns=calc.columns)


# ---------------------------------------------------------------------------
# Feature matrix builder
# ---------------------------------------------------------------------------
def build_feature_matrix(
    smiles: List[str],
    fingerprint_configs: Optional[List[dict]] = None,
    rdkit_descriptors: Optional[List[str]] = None,
    mordred_descriptors: Optional[List[str]] = None,
    fixed_feature_names: Optional[List[str]] = None,
    verbose: bool = True,
):
    """Build feature matrix with group indices for feature selection.

    Each fingerprint type is an atomic group. Each individual descriptor
    is its own group.

    Args:
        smiles: List of SMILES strings.
        fingerprint_configs: List of fingerprint config dicts with 'kind' key.
        rdkit_descriptors: List of RDKit descriptor names (individual groups).
        mordred_descriptors: List of Mordred descriptor names (individual groups).
        fixed_feature_names: Feature names to always include (not counted in k_features).
        verbose: Print progress.

    Returns:
        X: Feature matrix (n_samples, n_features).
        feature_groups: List of feature index groups.
        feature_names: List of group names.
        fixed_group_indices: List of group indices that are always included.
    """
    if fingerprint_configs is None:
        fingerprint_configs = []
    if fixed_feature_names is None:
        fixed_feature_names = []

    feature_matrices = []
    feature_groups = []
    feature_names = []
    fixed_group_indices = []
    current_idx = 0

    start_time = time.time()

    # Count groups for progress bar
    n_groups = len(fixed_feature_names)
    for fp in fingerprint_configs:
        if fp["kind"] not in fixed_feature_names:
            n_groups += 1
    has_descs = bool(rdkit_descriptors or mordred_descriptors)
    if has_descs:
        n_groups += 1

    pbar = tqdm(total=n_groups, desc="Computing features", unit="group",
                disable=not verbose, leave=False)

    # 1. Fixed features (fingerprints that are always included)
    for fixed_name in fixed_feature_names:
        fp_config = {"kind": fixed_name}
        for cfg in fingerprint_configs:
            if cfg.get("kind") == fixed_name:
                fp_config = cfg
                break
        transformer = FingerprintTransformer(**fp_config)
        fixed_X = transformer.fit_transform(smiles)
        feature_matrices.append(fixed_X)
        group_idx = len(feature_groups)
        feature_groups.append(list(range(current_idx, current_idx + fixed_X.shape[1])))
        feature_names.append(fixed_name)
        fixed_group_indices.append(group_idx)
        current_idx += fixed_X.shape[1]
        pbar.update(1)

    # 2. Fingerprints (each as atomic group, skip if already fixed)
    for fp_config in fingerprint_configs:
        fp_kind = fp_config["kind"]
        if fp_kind in fixed_feature_names:
            continue
        transformer = FingerprintTransformer(**fp_config)
        fp_X = transformer.fit_transform(smiles)
        feature_matrices.append(fp_X)
        feature_groups.append(list(range(current_idx, current_idx + fp_X.shape[1])))
        feature_names.append(fp_kind)
        current_idx += fp_X.shape[1]
        pbar.update(1)

    # 3. Descriptors (each as individual group)
    requested_descs = []
    if rdkit_descriptors:
        requested_descs.extend(rdkit_descriptors)
    if mordred_descriptors:
        requested_descs.extend(mordred_descriptors)

    if requested_descs:
        # Compute all descriptors
        rdkit_df = _compute_rdkit_descriptors(smiles)
        mordred_df = _compute_mordred_descriptors(smiles)
        all_desc_df = pd.concat([rdkit_df, mordred_df], axis=1)

        available_descs = [d for d in requested_descs if d in all_desc_df.columns]
        if available_descs:
            desc_df = all_desc_df[available_descs]
            feature_matrices.append(desc_df.values)
            for desc_name in desc_df.columns:
                feature_groups.append([current_idx])
                feature_names.append(desc_name)
                current_idx += 1
        pbar.update(1)

    pbar.close()

    X = np.hstack(feature_matrices) if feature_matrices else np.empty((len(smiles), 0))

    if verbose:
        elapsed = time.time() - start_time
        logger.info(
            f"Feature matrix: {X.shape} | {len(feature_groups)} groups | {elapsed:.1f}s"
        )

    return X, feature_groups, feature_names, fixed_group_indices

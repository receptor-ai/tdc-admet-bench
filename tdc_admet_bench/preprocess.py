"""SMILES preprocessing and standardisation."""

import logging
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
import datamol as dm
from rdkit.Chem import rdmolops

logger = logging.getLogger(__name__)

FILTER_FUNCTIONS = {
    "heavy_atoms": dm.descriptors.n_heavy_atoms,
    "rotatable_bonds": dm.descriptors.n_rotatable_bonds,
    "clogp": dm.descriptors.clogp,
    "hbd": dm.descriptors.n_hbd,
    "hba": dm.descriptors.n_hba,
    "molecular_weight": dm.descriptors.mw,
    "tpsa": dm.descriptors.tpsa,
}


def _parse_and_fix_smiles(smiles):
    mol = dm.to_mol(smiles)
    if mol is not None:
        return mol
    mol_unsanitized = dm.to_mol(smiles, sanitize=False)
    if mol_unsanitized is not None:
        try:
            return dm.fix_mol(mol_unsanitized)
        except Exception:
            return None
    return None


def _keep_largest_fragment(mol):
    if mol is None:
        return None
    fragments = rdmolops.GetMolFrags(mol, asMols=True)
    if not fragments:
        return mol
    return max(fragments, key=lambda m: (m.GetNumHeavyAtoms(), m.GetNumAtoms()))


def preprocess_smiles(
    smiles: List[str],
    keep_stereo: bool = False,
    keep_largest: bool = True,
    neutralize: bool = True,
    filters: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[pd.Series, np.ndarray]:
    """Standardise SMILES and apply property filters.

    Returns:
        (processed_smiles, kept_indices)
    """
    if filters is None:
        filters = []

    df = pd.DataFrame({"smiles": smiles, "original_index": np.arange(len(smiles))})

    with dm.without_rdkit_log():
        df["mol"] = df["smiles"].apply(_parse_and_fix_smiles)
        df = df[df["mol"].notna()].copy()

        if keep_largest:
            df["mol"] = df["mol"].apply(_keep_largest_fragment)

        df["mol"] = df["mol"].apply(
            dm.standardize_mol, disconnect_metals=True, uncharge=neutralize
        )

        if keep_largest:
            df["mol"] = df["mol"].apply(_keep_largest_fragment)

        for filter_dict in filters:
            for filter_name, filter_range in filter_dict.items():
                if filter_name not in FILTER_FUNCTIONS:
                    raise ValueError(f"Unknown filter: {filter_name}")
                descriptor_fn = FILTER_FUNCTIONS[filter_name]
                min_val, max_val = filter_range
                mask = df["mol"].apply(descriptor_fn).between(min_val, max_val)
                df = df[mask].copy()

    processed_smiles = df["mol"].apply(dm.to_smiles, isomeric=keep_stereo)
    kept_indices = df["original_index"].values
    return processed_smiles, kept_indices


def preprocess_dataset(
    df: pd.DataFrame,
    smiles_col: str = "smiles",
    target_col: Optional[str] = "target",
    aggregate_duplicates: Optional[str] = None,
    keep_stereo: bool = False,
    keep_largest: bool = True,
    neutralize: bool = True,
    filters: Optional[List[Dict[str, Any]]] = None,
) -> pd.DataFrame:
    """Preprocess DataFrame: standardise SMILES and optionally aggregate duplicates.

    Args:
        aggregate_duplicates: None, 'mean' (regression), or 'mode' (classification).
    """
    smiles_list = df[smiles_col].tolist()
    processed_smiles, kept_indices = preprocess_smiles(
        smiles_list,
        keep_stereo=keep_stereo,
        keep_largest=keep_largest,
        neutralize=neutralize,
        filters=filters,
    )

    result_data = {smiles_col: processed_smiles.tolist()}
    if target_col is not None:
        result_data[target_col] = df[target_col].iloc[kept_indices].values

    result_df = pd.DataFrame(result_data)

    if aggregate_duplicates and target_col is not None:
        if aggregate_duplicates == "mean":
            result_df = result_df.groupby(smiles_col, as_index=False)[target_col].mean()
        elif aggregate_duplicates == "mode":
            result_df = (
                result_df.groupby(smiles_col)[target_col]
                .agg(lambda x: x.mode()[0])
                .reset_index()
            )
        result_df = result_df.reset_index(drop=True)

    return result_df

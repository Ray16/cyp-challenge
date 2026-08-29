"""Local pre-flight checks for submission files, mirroring the validation done
by reference/space_app/submission.py (_read_tabular_submission / submit_predictions)
so mistakes are caught before spending one of the rate-limited upload attempts.

This does NOT reproduce the actual scoring (ST-RAE / MCC) — that runs server-side
in a Lambda not included in the Space source.
"""

import numpy as np
import pandas as pd

from . import config


def _check_identifiers(df: pd.DataFrame, test_df: pd.DataFrame) -> list[str]:
    errors = []
    if set(df["Molecule_Name"]) != set(test_df["Molecule_Name"]):
        errors.append("Molecule_Name values do not exactly match the blinded test set.")
    return errors


def validate_regression_submission(df: pd.DataFrame, test_df: pd.DataFrame) -> list[str]:
    errors = []
    missing = [c for c in config.REQUIRED_REGRESSION_COLUMNS if c not in df.columns]
    if missing:
        errors.append(f"Missing required columns: {missing}")
    if len(df) != config.ACTIVITY_DATASET_SIZE:
        errors.append(
            f"Expected {config.ACTIVITY_DATASET_SIZE} rows, got {len(df)}."
        )
    for col in config.REGRESSION_ENDPOINTS:
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        if values.isna().any() or np.isinf(values).any():
            errors.append(f"Column '{col}' contains NaN, inf, or non-numeric values.")
    errors.extend(_check_identifiers(df, test_df))
    return errors


def validate_classification_submission(df: pd.DataFrame, test_df: pd.DataFrame) -> list[str]:
    errors = []
    missing = [c for c in config.REQUIRED_CLASSIFICATION_COLUMNS if c not in df.columns]
    if missing:
        errors.append(f"Missing required columns: {missing}")
    if len(df) != config.ACTIVITY_DATASET_SIZE:
        errors.append(
            f"Expected {config.ACTIVITY_DATASET_SIZE} rows, got {len(df)}."
        )
    for col in config.CLASSIFICATION_ENDPOINTS:
        if col not in df.columns:
            continue
        unique_vals = set(pd.unique(df[col].dropna()))
        if not unique_vals <= {0, 1, True, False}:
            errors.append(f"Column '{col}' must be binary (0/1 or True/False), got {unique_vals}.")
    errors.extend(_check_identifiers(df, test_df))
    return errors

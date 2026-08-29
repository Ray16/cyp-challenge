"""Thin loaders for the raw challenge CSVs."""

from pathlib import Path

import pandas as pd

from . import config

ROOT = Path(__file__).resolve().parents[2]


def _load(relative_path: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / relative_path)


def load_train_inhibition() -> pd.DataFrame:
    return _load(config.TRAIN_INHIBITION_FILE)


def load_test_blinded() -> pd.DataFrame:
    return _load(config.TEST_FILE)


def load_train_tdi() -> pd.DataFrame:
    return _load(config.TRAIN_TDI_FILE)


def load_train_emax() -> pd.DataFrame:
    return _load(config.TRAIN_EMAX_FILE)


def load_train_single_concentration() -> pd.DataFrame:
    return _load(config.TRAIN_SINGLE_CONC_FILE)

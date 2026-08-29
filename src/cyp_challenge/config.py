"""Challenge constants, mirrored from reference/space_app/config.py.

Kept separate from that file (which is vendored, read-only reference code) so our
own pipeline code has a stable import path. If OpenADMET changes the schema,
diff against reference/space_app/config.py and update here.
"""

IDENTIFIER_COLUMNS = ["SMILES", "Molecule_Name"]

ISOFORMS = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]

REGRESSION_ENDPOINTS = [
    "CYP1A2_pIC50_direct_inhibition",
    "CYP2C9_pIC50_direct_inhibition",
    "CYP2D6_pIC50_direct_inhibition",
    "CYP3A4_pIC50_direct_inhibition",
]

CLASSIFICATION_ENDPOINTS = [
    "CYP2D6_is_TDI",
    "CYP3A4_is_TDI",
]

REQUIRED_REGRESSION_COLUMNS = IDENTIFIER_COLUMNS + REGRESSION_ENDPOINTS
REQUIRED_CLASSIFICATION_COLUMNS = IDENTIFIER_COLUMNS + CLASSIFICATION_ENDPOINTS

# Number of rows in cyp-challenge-TEST-BLINDED.csv; every submission file must have
# exactly this many rows.
ACTIVITY_DATASET_SIZE = 750

STRUCTURE_DATASET_SIZE = 184  # placeholder per upstream config, track not live yet

RAW_DATA_DIR = "data/raw"
TEST_FILE = f"{RAW_DATA_DIR}/cyp-challenge-TEST-BLINDED.csv"
TRAIN_INHIBITION_FILE = f"{RAW_DATA_DIR}/cyp-challenge-TRAIN_inhibition.csv"
TRAIN_TDI_FILE = f"{RAW_DATA_DIR}/cyp-challenge-TRAIN_TDI.csv"
TRAIN_EMAX_FILE = f"{RAW_DATA_DIR}/cyp-challenge-TRAIN_Emax.csv"
TRAIN_SINGLE_CONC_FILE = f"{RAW_DATA_DIR}/cyp-challenge-single-concentration-TRAIN.csv"

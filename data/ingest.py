"""
data/ingest.py

Responsible for:
1. Downloading the UCI Adult Income dataset (train + test splits)
2. Cleaning raw CSV — fixing column names, stripping whitespace, handling '?' missing values
3. Saving clean parquet files to the feature store directory (data/feature_store/)
4. Providing a load function that returns a clean DataFrame for use by features.py

Run directly to download and prepare data:
    python -m data.ingest

UCI dataset source:
    https://archive.ics.uci.edu/ml/machine-learning-databases/adult/
    - adult.data  → training split (32,561 rows)
    - adult.test  → test split    (16,281 rows)
"""

from __future__ import annotations

import logging
# import os
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Raw UCI download URLs
UCI_BASE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult"
TRAIN_URL    = f"{UCI_BASE_URL}/adult.data"
TEST_URL     = f"{UCI_BASE_URL}/adult.test"

# Local paths
RAW_DIR          = Path("data/raw")
FEATURE_STORE    = Path("data/feature_store")
RAW_TRAIN_CSV    = RAW_DIR / "adult_train.csv"
RAW_TEST_CSV     = RAW_DIR / "adult_test.csv"
CLEAN_TRAIN_FILE = FEATURE_STORE / "train.parquet"
CLEAN_TEST_FILE  = FEATURE_STORE / "test.parquet"

# Column names — UCI dataset ships with no header row
COLUMN_NAMES = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",               # target: ">50K" or "<=50K"
]

# Target label mapping — normalise to clean strings without trailing periods
# The test split has labels like ">50K." (with a dot) — we strip that.
LABEL_MAP = {
    ">50K" : ">50K",
    ">50K.": ">50K",
    "<=50K" : "<=50K",
    "<=50K.": "<=50K",
}


# ---------------------------------------------------------------------------
# Step 1 — Download raw CSVs
# ---------------------------------------------------------------------------

def _download_file(url: str, dest: Path) -> None:
    """Download a file from url to dest, skipping if already present."""
    if dest.exists():
        log.info("Already downloaded: %s — skipping.", dest)
        return

    log.info("Downloading %s → %s", url, dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(url, timeout=30)
    response.raise_for_status()

    dest.write_bytes(response.content)
    log.info("Saved %d bytes to %s", len(response.content), dest)


def download_raw_data() -> None:
    """Download both train and test splits from UCI."""
    _download_file(TRAIN_URL, RAW_TRAIN_CSV)
    _download_file(TEST_URL,  RAW_TEST_CSV)


# ---------------------------------------------------------------------------
# Step 2 — Load and clean raw CSV
# ---------------------------------------------------------------------------

def _load_raw_csv(path: Path, skip_rows: int = 0) -> pd.DataFrame:
    """
    Load a raw UCI CSV into a DataFrame.

    skip_rows=1 for the test file because adult.test has a comment on line 1:
    '|1x3 Cross validator'
    """
    df = pd.read_csv(
        path,
        names=COLUMN_NAMES,
        skiprows=skip_rows,
        skipinitialspace=True,   # UCI values have leading spaces e.g. " Private"
        na_values="?",           # UCI encodes missing values as "?"
    )
    return df


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all cleaning steps to a raw UCI DataFrame.

    Steps
    -----
    1. Strip whitespace from all string columns (UCI has " Private", " Male", etc.)
    2. Normalise income labels — test split has trailing "." e.g. ">50K."
    3. Drop rows where any categorical feature is missing (NaN from "?")
       These are ~2% of rows and dropping is standard for this dataset.
    4. Reset index after dropping.
    """
    # 1. Strip whitespace from object columns
    str_cols = df.select_dtypes(include="object").columns
    df[str_cols] = df[str_cols].apply(lambda col: col.str.strip())

    # 2. Normalise income labels
    df["income"] = df["income"].map(LABEL_MAP)

    # 3. Drop rows with any missing categorical value
    before = len(df)
    df = df.dropna()
    after  = len(df)
    log.info("Dropped %d rows with missing values (%.1f%%)", before - after, 100 * (before - after) / before)

    # 4. Reset index
    df = df.reset_index(drop=True)

    return df


def load_and_clean(split: str = "train") -> pd.DataFrame:
    """
    Load raw CSV, apply cleaning, and return a clean DataFrame.

    Parameters
    ----------
    split : "train" | "test"

    Returns
    -------
    pd.DataFrame with COLUMN_NAMES columns, no missing values, clean labels.
    """
    if split == "train":
        df = _load_raw_csv(RAW_TRAIN_CSV, skip_rows=0)
    elif split == "test":
        df = _load_raw_csv(RAW_TEST_CSV, skip_rows=1)   # skip the comment line
    else:
        raise ValueError(f"split must be 'train' or 'test', got '{split}'")

    df = _clean(df)
    log.info("Loaded %s split: %d rows × %d cols", split, *df.shape)
    return df


# ---------------------------------------------------------------------------
# Step 3 — Save to feature store as Parquet
# ---------------------------------------------------------------------------

def save_to_feature_store(df: pd.DataFrame, split: str) -> Path:
    """
    Persist a clean DataFrame to the feature store as Parquet.

    Parquet is chosen over CSV because:
    - Stores dtypes — integers stay integers, no silent string conversion
    - ~5x smaller on disk due to columnar compression
    - Faster to read back (columnar scan)
    - Industry standard for feature stores (S3 + Parquet is the de facto pattern)
    """
    FEATURE_STORE.mkdir(parents=True, exist_ok=True)
    dest = CLEAN_TRAIN_FILE if split == "train" else CLEAN_TEST_FILE
    df.to_parquet(dest, index=False, engine="pyarrow")
    log.info("Saved %s split to feature store: %s", split, dest)
    return dest


# ---------------------------------------------------------------------------
# Step 4 — Public loader (used by features.py and train.py)
# ---------------------------------------------------------------------------

def load_from_feature_store(split: str = "train") -> pd.DataFrame:
    """
    Load a clean, parquet-stored DataFrame from the feature store.

    This is the function imported by:
      - data/features.py  (feature engineering)
      - training/train.py (model training)

    Parameters
    ----------
    split : "train" | "test"
    """
    path = CLEAN_TRAIN_FILE if split == "train" else CLEAN_TEST_FILE

    if not path.exists():
        raise FileNotFoundError(
            f"Feature store file not found: {path}\n"
            "Run `python -m data.ingest` first to download and prepare the data."
        )

    df = pd.read_parquet(path, engine="pyarrow")
    log.info("Loaded %s split from feature store: %d rows × %d cols", split, *df.shape)
    return df


# ---------------------------------------------------------------------------
# Step 5 — Basic sanity checks after loading
# ---------------------------------------------------------------------------

def validate_schema(df: pd.DataFrame) -> None:
    """
    Assert that the DataFrame has the expected columns and no missing values.
    Raises AssertionError if anything is wrong — caught by the Airflow DAG.
    """
    missing_cols = set(COLUMN_NAMES) - set(df.columns)
    assert not missing_cols, f"Missing columns after ingestion: {missing_cols}"

    null_counts = df.isnull().sum()
    assert null_counts.sum() == 0, f"Null values found after cleaning:\n{null_counts[null_counts > 0]}"

    assert set(df["income"].unique()) == {">50K", "<=50K"}, (
        f"Unexpected income labels: {df['income'].unique()}"
    )

    log.info("Schema validation passed ✓")


# ---------------------------------------------------------------------------
# Entrypoint — run as script to download + prepare data
# ---------------------------------------------------------------------------

def run_ingestion_pipeline() -> None:
    """Full pipeline: download → clean → validate → save to feature store."""
    log.info("=" * 60)
    log.info("Starting data ingestion pipeline")
    log.info("=" * 60)

    # Download
    download_raw_data()

    # Train split
    train_df = load_and_clean("train")
    validate_schema(train_df)
    save_to_feature_store(train_df, "train")

    # Test split
    test_df = load_and_clean("test")
    validate_schema(test_df)
    save_to_feature_store(test_df, "test")

    log.info("=" * 60)
    log.info(
        "Ingestion complete. Train: %d rows | Test: %d rows",
        len(train_df), len(test_df),
    )
    log.info("Feature store location: %s", FEATURE_STORE.resolve())
    log.info("=" * 60)


if __name__ == "__main__":
    run_ingestion_pipeline()
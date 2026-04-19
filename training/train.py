# =============================================================================
# training/train.py
# =============================================================================
# PURPOSE:
#   This is the central training script. It orchestrates the full model
#   training pipeline:
#     1. Pull raw data via ingest.py
#     2. Validate it via validate.py
#     3. Build features via features.py
#     4. Train a scikit-learn model (GradientBoostingClassifier by default)
#     5. Save the trained model + feature pipeline as versioned artifacts
#
#   This script is called by:
#     - airflow/dags/training_dag.py  (scheduled / triggered retraining)
#     - CLI: `python -m training.train` for local runs
#
#   Outputs (written to MODEL_DIR defined in .env):
#     - model_v{timestamp}.pkl        — the trained sklearn model
#     - pipeline_v{timestamp}.pkl     — the fitted FeaturePipeline
#     - train_meta_v{timestamp}.json  — run metadata (params, data stats, etc.)
# =============================================================================

import os
import json
import logging
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

# --- Internal imports from our own pipeline modules ---
# ingest.py  → pulls raw DataFrame from DB / CSV / S3
# validate.py → checks data quality before training
# features.py → fits + transforms features; returns X ready for sklearn
from data.ingest import run_ingestion_pipeline
from data.validate import validate_dataframe
from data.features import build_features, FeaturePipeline
# from api.schemas import FEATURE_SCHEMA  # column contract / target name
from api.schemas import PredictionInput, PredictionOutput  # for type hints and contract validation

# =============================================================================
# LOGGING SETUP
# =============================================================================
# Using Python's standard logging so every module in the project writes
# structured logs in the same format. Airflow also captures these logs.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("training.train")


# =============================================================================
# CONSTANTS / CONFIGURATION
# =============================================================================
# These values are read from environment variables so they can be overridden
# in docker-compose.yml or the Airflow DAG without changing code.

# Directory where trained model artifacts are saved
MODEL_DIR = Path(os.getenv("MODEL_DIR", "artifacts/models"))

# Name of the target column (what we're predicting)
# Defined once in schemas.py so all modules agree on the same column name
TARGET_COL = "income"

# Fraction of data held out for a quick in-training validation split
# (evaluate.py does the full eval later on a separate held-out set)
VALIDATION_SPLIT = float(os.getenv("TRAIN_VAL_SPLIT", "0.2"))

# Random seed for reproducibility across training runs
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))

# Hyperparameters for GradientBoostingClassifier — read from env so we can
# tune them from the Airflow DAG or CI without touching this file
DEFAULT_PARAMS = {
    "n_estimators":   int(os.getenv("GBM_N_ESTIMATORS", "200")),
    "max_depth":      int(os.getenv("GBM_MAX_DEPTH", "4")),
    "learning_rate":  float(os.getenv("GBM_LEARNING_RATE", "0.05")),
    "subsample":      float(os.getenv("GBM_SUBSAMPLE", "0.8")),
    "random_state":   RANDOM_SEED,
}


# =============================================================================
# ARTIFACT HELPERS
# =============================================================================

def _make_version_tag() -> str:
    """
    Creates a timestamp-based version string, e.g. '20240415_143022'.
    Every training run gets a unique tag so artifacts never overwrite each other.
    This makes it easy to roll back to a previous model version.
    """
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def _ensure_model_dir() -> None:
    """
    Creates MODEL_DIR (and any parent dirs) if it doesn't already exist.
    Safe to call multiple times — exist_ok=True prevents errors.
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Model artifact directory: %s", MODEL_DIR.resolve())


def save_model(model, version_tag: str) -> Path:
    """
    Serializes the trained sklearn model to disk using pickle.

    Args:
        model:       A fitted sklearn estimator (e.g. GradientBoostingClassifier)
        version_tag: Unique string suffix, e.g. '20240415_143022'

    Returns:
        Path to the saved .pkl file
    """
    path = MODEL_DIR / f"model_v{version_tag}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info("Model saved → %s", path)
    return path


def save_pipeline(pipeline: FeaturePipeline, version_tag: str) -> Path:
    """
    Saves the FITTED FeaturePipeline alongside the model.

    Why save the pipeline?
      At prediction time (predict.py) we must apply the EXACT same
      transformations that were used during training — same imputation values,
      same scaler means/stds, same one-hot encoder categories.
      Saving the fitted pipeline as an artifact guarantees this.

    Args:
        pipeline:    A fitted FeaturePipeline instance
        version_tag: Matches the model's version tag so they stay paired

    Returns:
        Path to the saved pipeline .pkl file
    """
    path = MODEL_DIR / f"pipeline_v{version_tag}.pkl"
    pipeline.save(str(path))          # FeaturePipeline.save() uses joblib internally
    logger.info("Feature pipeline saved → %s", path)
    return path


def save_metadata(meta: dict, version_tag: str) -> Path:
    """
    Writes a JSON file with everything needed to reproduce or audit this run:
      - training timestamp
      - data row counts
      - hyperparameters
      - train/val split sizes
      - validation score (accuracy on the in-training val split)

    This metadata is later read by register_model.py to decide whether to
    promote this model to 'production'.

    Args:
        meta:        Dictionary of key/value training info
        version_tag: Matches model + pipeline version tag

    Returns:
        Path to the saved .json file
    """
    path = MODEL_DIR / f"train_meta_v{version_tag}.json"
    with open(path, "w") as f:
        json.dump(meta, f, indent=2, default=str)   # default=str handles datetime objects
    logger.info("Training metadata saved → %s", path)
    return path


# =============================================================================
# CORE TRAINING LOGIC
# =============================================================================

def train(
    model_params: Optional[dict] = None,
    data_source: Optional[str] = None,
) -> dict:
    """
    End-to-end training run. Called by the Airflow DAG or CLI.

    Pipeline steps:
      1. Ingest raw data
      2. Validate data quality (hard stop if critical checks fail)
      3. Build features (fit + transform)
      4. Split into train / val
      5. Fit the model
      6. Quick val-set accuracy check
      7. Save model, pipeline, metadata
      8. Return a result dict (used by register_model.py)

    Args:
        model_params:  Optional hyperparameter overrides (dict).
                       Merged on top of DEFAULT_PARAMS — useful for HPO.
        data_source:   Optional override for ingest source (path / table name).
                       If None, ingest.py uses its own env-var config.

    Returns:
        A dict with keys: version_tag, model_path, pipeline_path,
                          meta_path, val_accuracy, n_train, n_val
    """

    # -------------------------------------------------------------------------
    # STEP 0 — Setup
    # -------------------------------------------------------------------------
    version_tag = _make_version_tag()
    _ensure_model_dir()
    params = {**DEFAULT_PARAMS, **(model_params or {})}  # merge any overrides

    logger.info("=" * 60)
    logger.info("Training run started  |  version: %s", version_tag)
    logger.info("Hyperparameters: %s", params)

    # -------------------------------------------------------------------------
    # STEP 1 — Ingest raw data
    # -------------------------------------------------------------------------
    # ingest_data() connects to the configured source (CSV / DB / S3) and
    # returns a raw pd.DataFrame before any transformation.
    logger.info("Step 1/7 — Ingesting data …")
    t0 = time.perf_counter()

    raw_df = run_ingestion_pipeline()# source=None → uses env config
    # fixed the error
    import pandas as pd

    train_path = "data/feature_store/train.parquet"
    raw_df = pd.read_parquet(train_path)

    logger.info(
    "Loaded training data: %d rows × %d cols",
    len(raw_df),
    raw_df.shape[1],
    )

    logger.info(
        "Ingested %d rows × %d cols in %.2fs",
        len(raw_df), raw_df.shape[1], time.perf_counter() - t0,
    )

    # -------------------------------------------------------------------------
    # STEP 2 — Validate data quality
    # -------------------------------------------------------------------------
    # validate_dataframe() runs hard + soft checks defined in validate.py.
    # If any HARD check fails (schema mismatch, too many nulls, etc.) we raise
    # immediately — a bad model trained on bad data is worse than no model.
    logger.info("Step 2/7 — Validating data …")

    report = validate_dataframe(raw_df)

    if not report.passed:
        # Log every failing check so the engineer knows exactly what's wrong
        for check in report.errors:
            logger.error("  HARD CHECK FAILED: %s — %s", check.check, check.detail)
        raise ValueError(
            f"Data validation failed ({len(report.errors)} hard checks). "
            "Aborting training. Fix the data issues listed above."
        )

    # Soft check failures are logged as warnings but don't stop training
    for warning in report.warnings:
        logger.warning("  Soft check warning: %s", warning)

    logger.info("Data validation passed  (%d rows clean)", report.row_count)

    # -------------------------------------------------------------------------
    # STEP 3 — Feature engineering
    # -------------------------------------------------------------------------
    # build_features(df, fit=True):
    #   - fit=True  → fits the FeaturePipeline on this data, then transforms
    #   - Returns (X: np.ndarray, y: np.ndarray, pipeline: FeaturePipeline)
    #
    # We keep the pipeline object so we can save it as an artifact.
    # At inference time predict.py calls pipeline.transform() — never fit().
    logger.info("Step 3/7 — Building features (fit=True) …")

    X, y, pipeline = build_features(raw_df, fit=True)

     # build_features() returns the fitted pipeline as well
   

    logger.info(
        "Feature matrix shape: %s  |  target distribution: %s",
        X.shape,
        dict(zip(*np.unique(y, return_counts=True))),
    )

    # -------------------------------------------------------------------------
    # STEP 4 — Train / validation split
    # -------------------------------------------------------------------------
    # We do a simple stratified split here so train + val have the same class
    # balance. This split is NOT the official evaluation split — that happens
    # in evaluate.py with a held-out test set that never touches training.
    logger.info("Step 4/7 — Splitting data (val=%.0f%%) …", VALIDATION_SPLIT * 100)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=VALIDATION_SPLIT,
        random_state=RANDOM_SEED,
        stratify=y,           # preserves class ratio in both splits
    )

    logger.info(
        "Split sizes — train: %d  |  val: %d", len(y_train), len(y_val)
    )

    # -------------------------------------------------------------------------
    # STEP 5 — Model training
    # -------------------------------------------------------------------------
    # GradientBoostingClassifier is our default model. It works well for
    # tabular data without heavy tuning and gives calibrated probabilities.
    #
    # To swap in a different model (XGBoost, LightGBM, etc.) you'd:
    #   1. Change the class instantiation here
    #   2. Update DEFAULT_PARAMS with the new model's hyperparameter names
    #   3. Everything else in the pipeline stays the same
    logger.info("Step 5/7 — Training model …")
    t1 = time.perf_counter()

    model = GradientBoostingClassifier(**params)
    model.fit(X_train, y_train)

    train_duration = time.perf_counter() - t1
    logger.info("Model trained in %.2fs", train_duration)

    # -------------------------------------------------------------------------
    # STEP 6 — Quick validation accuracy
    # -------------------------------------------------------------------------
    # This is a fast sanity check — not the official model evaluation.
    # A very low val_accuracy here signals something went badly wrong
    # (e.g. feature pipeline bug, label leakage gone, etc.)
    logger.info("Step 6/7 — Evaluating on val split …")

    val_preds = model.predict(X_val)
    val_accuracy = float(np.mean(val_preds == y_val))

    logger.info("Val accuracy: %.4f", val_accuracy)

    # Hard stop if accuracy is suspiciously low — saves a garbage model
    MIN_ACCEPTABLE_ACCURACY = float(os.getenv("MIN_TRAIN_ACCURACY", "0.5"))
    if val_accuracy < MIN_ACCEPTABLE_ACCURACY:
        raise ValueError(
            f"Val accuracy {val_accuracy:.4f} is below minimum threshold "
            f"{MIN_ACCEPTABLE_ACCURACY}. Something is wrong — aborting save."
        )

    # -------------------------------------------------------------------------
    # STEP 7 — Save artifacts
    # -------------------------------------------------------------------------
    logger.info("Step 7/7 — Saving artifacts …")

    model_path    = save_model(model, version_tag)
    pipeline_path = save_pipeline(pipeline, version_tag)

    # Build metadata dict — everything register_model.py needs to make a
    # promotion decision (accuracy, data size, params, timing)
    meta = {
        "version_tag":      version_tag,
        "trained_at":       datetime.utcnow().isoformat(),
        "n_rows_ingested":  len(raw_df),
        "n_rows_validated": report.row_count,
        "n_train":          int(len(y_train)),
        "n_val":            int(len(y_val)),
        "n_features":       int(X.shape[1]),
        "val_accuracy":     round(val_accuracy, 6),
        "train_duration_s": round(train_duration, 3),
        "hyperparameters":  params,
        "model_path":       str(model_path),
        "pipeline_path":    str(pipeline_path),
        "target_column":    TARGET_COL,
        "random_seed":      RANDOM_SEED,
        "soft_check_warnings": [
            {"message": warning}
            for warning in report.warnings
            ],
    }

    meta_path = save_metadata(meta, version_tag)

    logger.info("=" * 60)
    logger.info("Training run COMPLETE  |  version: %s  |  val_acc: %.4f",
                version_tag, val_accuracy)
    logger.info("=" * 60)

    # Return a result dict so the Airflow DAG or register_model.py can
    # use XCom / direct call to pick up paths and metrics
    return {
        "version_tag":    version_tag,
        "model_path":     str(model_path),
        "pipeline_path":  str(pipeline_path),
        "meta_path":      str(meta_path),
        "val_accuracy":   val_accuracy,
        "n_train":        int(len(y_train)),
        "n_val":          int(len(y_val)),
    }


# =============================================================================
# CLI ENTRY POINT
# =============================================================================
# Allows running this script directly:
#   python -m training.train
#   python -m training.train --n_estimators 300 --max_depth 5
#
# The Airflow DAG calls train() programmatically, so argparse is only for
# local developer runs and manual retraining.
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the ML model from scratch.")
    parser.add_argument("--n_estimators", type=int,   default=None, help="Number of boosting rounds")
    parser.add_argument("--max_depth",    type=int,   default=None, help="Max tree depth")
    parser.add_argument("--learning_rate",type=float, default=None, help="Shrinkage factor")
    parser.add_argument("--subsample",    type=float, default=None, help="Row subsampling fraction")
    parser.add_argument("--data_source",  type=str,   default=None, help="Override ingest source path/table")
    args = parser.parse_args()

    # Build param overrides from CLI — only include args that were actually set
    overrides = {
        k: v for k, v in vars(args).items()
        if v is not None and k != "data_source"
    }

    result = train(model_params=overrides or None, data_source=args.data_source)

    print("\n--- Training Result ---")
    for k, v in result.items():
        print(f"  {k}: {v}")
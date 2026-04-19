"""
data/features.py
----------------
Feature engineering pipeline.

Takes a validated raw DataFrame (output of data/validate.py) and produces
a model-ready feature matrix.

Steps
-----
1.  Type casting          – enforce correct dtypes from schemas.py
2.  Temporal features     – extract hour, day-of-week, is_weekend from timestamp
3.  Numeric transforms    – log1p on skewed columns, standard scaling
4.  Categorical encoding  – ordinal encoding for known categoricals
5.  Interaction features  – domain-driven cross features
6.  Missing value impute  – median for numeric, mode for categorical
7.  Feature selection     – drop raw/redundant columns not needed by model

All transforms are stateful (fitted on training data, applied on inference).
Use ``FeaturePipeline.fit_transform(df)`` during training and
``FeaturePipeline.transform(df)`` at inference time.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Columns that will be log1p-transformed before scaling (right-skewed)
LOG_TRANSFORM_COLS: list[str] = [
    "capital_gain",
    "capital_loss",
]

# Numeric columns to standard-scale (after log transform where applicable)
SCALE_COLS: list[str] = [
    "age",
    "fnlwgt",
    "education_num",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
]

# Categorical columns to ordinal-encode
CATEGORICAL_COLS: list[str] = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]

# Columns to drop before returning the feature matrix
DROP_COLS: list[str] = [
    "transaction_id",   # identifier, not a feature
    "customer_id",      # high-cardinality ID
    "timestamp",        # replaced by temporal features
    "raw_notes",        # free text — not used
]

# Timestamp column name
TIMESTAMP_COL: str = "timestamp"

# Default path to persist pipeline state
DEFAULT_PIPELINE_PATH: Path = Path("artifacts/feature_pipeline.pkl")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cast_types(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure correct dtypes so downstream transforms never break."""
    df = df.copy()

    # Parse timestamp
    if TIMESTAMP_COL in df.columns:
        df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], utc=True, errors="coerce")

    # Numeric coercion
    for col in SCALE_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # String coercion for categoricals
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str)

    return df


def _extract_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive hour, day-of-week, is_weekend, days_since_epoch from timestamp."""
    if TIMESTAMP_COL not in df.columns:
        logger.warning("'%s' column not found; skipping temporal features.", TIMESTAMP_COL)
        return df

    ts = df[TIMESTAMP_COL]

    df["hour_of_day"] = ts.dt.hour.astype(np.int8)
    df["day_of_week"] = ts.dt.dayofweek.astype(np.int8)   # 0=Mon … 6=Sun
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(np.int8)
    df["month"] = ts.dt.month.astype(np.int8)

    # Days since Unix epoch — captures broad time trends
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    df["days_since_epoch"] = (ts - epoch).dt.days.astype(np.int32)

    return df


def _add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Domain-driven cross features."""
    # amount per credit-score bucket — higher amount at lower score = riskier
    if "amount" in df.columns and "credit_score" in df.columns:
        # Use raw (pre-scaled) columns; they are still available at this point
        with np.errstate(divide="ignore", invalid="ignore"):
            df["amount_per_credit_score"] = np.where(
                df["credit_score"] > 0,
                df["amount"] / df["credit_score"],
                0.0,
            )

    # Transaction velocity flag — many txns in 30d AND large amount
    if "num_transactions_30d" in df.columns and "amount" in df.columns:
        df["high_velocity_large_amount"] = (
            (df["num_transactions_30d"] > df["num_transactions_30d"].median()) &
            (df["amount"] > df["amount"].median())
        ).astype(np.int8)

    return df


def _impute(df: pd.DataFrame) -> pd.DataFrame:
    """Simple median / mode imputation — fit-free, applied per batch."""
    for col in SCALE_COLS:
        if col in df.columns and df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.debug("Imputed '%s' with median=%.4f", col, median_val)

    for col in CATEGORICAL_COLS:
        if col in df.columns and df[col].isna().any():
            mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else "unknown"
            df[col] = df[col].fillna(mode_val)
            logger.debug("Imputed '%s' with mode='%s'", col, mode_val)

    return df


# ---------------------------------------------------------------------------
# FeaturePipeline
# ---------------------------------------------------------------------------

class FeaturePipeline:
    """
    Stateful feature engineering pipeline.

    Usage
    -----
    Training::

        pipeline = FeaturePipeline()
        X_train = pipeline.fit_transform(df_train)
        pipeline.save()

    Inference::

        pipeline = FeaturePipeline.load()
        X_inference = pipeline.transform(df_new)
    """

    def __init__(self) -> None:
        self._scaler: Optional[StandardScaler] = None
        self._encoder: Optional[OrdinalEncoder] = None
        self._scale_cols_present: list[str] = []
        self._cat_cols_present: list[str] = []
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Core transform steps (order matters)
    # ------------------------------------------------------------------

    def _base_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Steps that don't require fitted state."""
        df = _cast_types(df)
        df = _impute(df)
        df = _extract_temporal_features(df)
        df = _add_interaction_features(df)

        # Log-transform skewed columns BEFORE scaling
        for col in LOG_TRANSFORM_COLS:
            if col in df.columns:
                df[col] = np.log1p(df[col].clip(lower=0))

        return df

    def _drop_unused(self, df: pd.DataFrame) -> pd.DataFrame:
        cols_to_drop = [c for c in DROP_COLS if c in df.columns]
        return df.drop(columns=cols_to_drop)

    # ------------------------------------------------------------------
    # fit_transform  (training only)
    # ------------------------------------------------------------------

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the pipeline on *df* and return the transformed feature matrix.
        Call this only on training data.
        """
        logger.info("Fitting FeaturePipeline on %d rows.", len(df))
        df = self._base_transform(df)

        # Fit + apply scaler
        self._scale_cols_present = [c for c in SCALE_COLS if c in df.columns]
        self._scaler = StandardScaler()
        df[self._scale_cols_present] = self._scaler.fit_transform(
            df[self._scale_cols_present]
        )

        # Fit + apply encoder
        self._cat_cols_present = [c for c in CATEGORICAL_COLS if c in df.columns]
        self._encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )
        df[self._cat_cols_present] = self._encoder.fit_transform(
            df[self._cat_cols_present]
        )

        df = self._drop_unused(df)
        self._is_fitted = True

        logger.info(
            "FeaturePipeline fitted. Output shape: %s. Columns: %s",
            df.shape,
            df.columns.tolist(),
        )
        return df

    # ------------------------------------------------------------------
    # transform  (inference / evaluation)
    # ------------------------------------------------------------------

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the already-fitted pipeline to *df*.
        Raises ``RuntimeError`` if called before ``fit_transform``.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "FeaturePipeline has not been fitted. "
                "Call fit_transform() on training data first, or load() a saved pipeline."
            )

        df = self._base_transform(df)

        # Apply (don't re-fit) scaler
        present = [c for c in self._scale_cols_present if c in df.columns]
        if present:
            df[present] = self._scaler.transform(df[present])  # type: ignore[union-attr]

        # Apply (don't re-fit) encoder
        cat_present = [c for c in self._cat_cols_present if c in df.columns]
        if cat_present:
            df[cat_present] = self._encoder.transform(df[cat_present])  # type: ignore[union-attr]

        df = self._drop_unused(df)
        return df

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path | str = DEFAULT_PIPELINE_PATH) -> None:
        """Pickle the fitted pipeline to *path*."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("FeaturePipeline saved to %s", path)

    @classmethod
    def load(cls, path: Path | str = DEFAULT_PIPELINE_PATH) -> "FeaturePipeline":
        """Load a previously saved pipeline from *path*."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No pipeline found at '{path}'.")
        with open(path, "rb") as f:
            pipeline: FeaturePipeline = pickle.load(f)
        logger.info("FeaturePipeline loaded from %s", path)
        return pipeline

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def feature_names(self) -> list[str]:
        """Return the list of output feature column names after transform."""
        if not self._is_fitted:
            raise RuntimeError("Pipeline not fitted yet.")
        return self._scale_cols_present + self._cat_cols_present + [
            "hour_of_day",
            "day_of_week",
            "is_weekend",
            "month",
            "days_since_epoch",
            "amount_per_credit_score",
            "high_velocity_large_amount",
        ]


# ---------------------------------------------------------------------------
# Convenience function  (used by training/train.py & data/ingest.py)
# ---------------------------------------------------------------------------

# def build_features(
#     df: pd.DataFrame,
#     pipeline: Optional[FeaturePipeline] = None,
#     *,
#     fit: bool = False,
# ) -> tuple[pd.DataFrame, FeaturePipeline]:
#     """
#     High-level helper.

#     Parameters
#     ----------
#     df:
#         Validated raw DataFrame.
#     pipeline:
#         An existing ``FeaturePipeline`` instance. If None and ``fit=True``,
#         a new one is created and fitted.
#     fit:
#         If True, call ``fit_transform``; otherwise call ``transform``.

#     Returns
#     -------
#     (transformed_df, pipeline)
#     """
#     if pipeline is None:
#         pipeline = FeaturePipeline()

#     if fit:
#         transformed = pipeline.fit_transform(df)
#     else:
#         transformed = pipeline.transform(df)

#     return transformed, pipeline
# def build_features(
#     df: pd.DataFrame,
#     pipeline: Optional[FeaturePipeline] = None,
#     *,
#     fit: bool = False,
# ) -> tuple[pd.DataFrame, pd.Series, FeaturePipeline]:
#     """
#     High-level helper.

#     Returns:
#     -------
#     (X, y, pipeline)
#     """

#     TARGET_COL = "income"

#     if pipeline is None:
#         pipeline = FeaturePipeline()

#     # Separate target BEFORE transformation
#     y = df[TARGET_COL].copy()
#     X_df = df.drop(columns=[TARGET_COL])

#     if fit:
#         transformed = pipeline.fit_transform(X_df)
#     else:
#         transformed = pipeline.transform(X_df)

#     return transformed, y, pipeline
def build_features(
    df: pd.DataFrame,
    pipeline: Optional[FeaturePipeline] = None,
    *,
    fit: bool = False,
) -> tuple[pd.DataFrame, pd.Series, FeaturePipeline]:
    """
    High-level helper.

    Parameters
    ----------
    df:
        Validated raw DataFrame.
    pipeline:
        Existing FeaturePipeline (optional).
    fit:
        If True → fit + transform
        If False → only transform

    Returns
    -------
    (X, y, pipeline)
    """

    TARGET_COL = "income"

    if pipeline is None:
        pipeline = FeaturePipeline()

    # ------------------------------------------------------------------
    # Ensure target column exists
    # ------------------------------------------------------------------
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataframe.")

    # ------------------------------------------------------------------
    # Separate target BEFORE feature pipeline (prevents data leakage)
    # ------------------------------------------------------------------
    y = df[TARGET_COL].copy()
    X_df = df.drop(columns=[TARGET_COL])

    # ------------------------------------------------------------------
    # Apply pipeline ONLY on feature columns
    # ------------------------------------------------------------------
    if fit:
        transformed = pipeline.fit_transform(X_df)
    else:
        transformed = pipeline.transform(X_df)

    # ------------------------------------------------------------------
    # Ensure numeric array output for sklearn compatibility
    # ------------------------------------------------------------------
    if isinstance(transformed, pd.DataFrame):
        transformed = transformed.values

    return transformed, y, pipeline
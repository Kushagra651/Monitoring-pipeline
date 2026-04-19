"""
data/drift_injector.py
----------------------
Simulates realistic data drift scenarios for testing the monitoring pipeline.

Drift types supported
---------------------
1.  Covariate drift      – input feature distribution shifts (gradual / sudden)
2.  Label drift          – target class distribution shifts
3.  Concept drift        – relationship between features and label changes
4.  Missing value drift  – sudden spike in nulls for specific columns
5.  Schema drift         – unexpected new / dropped / renamed columns
6.  Categorical drift    – new unseen category labels appear
7.  Temporal drift       – timestamp gaps / out-of-order events

Each injector is a pure function:
    inject_*(df, **params) -> pd.DataFrame

A high-level ``inject_drift`` dispatcher accepts a ``DriftConfig`` dataclass
so Airflow DAGs and tests can drive injection declaratively.
"""

from __future__ import annotations

import logging
# import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enums & Config
# ---------------------------------------------------------------------------

class DriftType(str, Enum):
    COVARIATE        = "covariate"
    LABEL            = "label"
    CONCEPT          = "concept"
    MISSING_VALUE    = "missing_value"
    SCHEMA           = "schema"
    CATEGORICAL      = "categorical"
    TEMPORAL         = "temporal"
    NONE             = "none"           # pass-through (useful in A/B tests)


@dataclass
class DriftConfig:
    """
    Declarative configuration for a single drift injection run.

    Parameters
    ----------
    drift_type:
        Which drift scenario to apply.
    intensity:
        0.0 = no drift, 1.0 = maximum drift. Interpreted per injector.
    affected_columns:
        Columns to target. Empty list = injector picks sensible defaults.
    gradual:
        If True, drift is applied progressively across rows (simulates
        concept / covariate drift that worsens over time).
    seed:
        Random seed for reproducibility.
    extra:
        Injector-specific kwargs passed through as-is.
    """
    drift_type: DriftType = DriftType.NONE
    intensity: float = 0.3              # fraction of rows / magnitude
    affected_columns: list[str] = field(default_factory=list)
    gradual: bool = False
    seed: int = 42
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Default column sets (fallbacks when affected_columns is empty)
# ---------------------------------------------------------------------------

_DEFAULT_NUMERIC_COLS  = ["amount", "age", "credit_score", "num_transactions_30d"]
_DEFAULT_CAT_COLS      = ["currency", "channel", "account_type"]
_LABEL_COL             = "label"
_TIMESTAMP_COL         = "timestamp"


# ---------------------------------------------------------------------------
# 1. Covariate drift
# ---------------------------------------------------------------------------

def inject_covariate_drift(
    df: pd.DataFrame,
    *,
    intensity: float = 0.3,
    affected_columns: Optional[list[str]] = None,
    gradual: bool = False,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Shift numeric feature distributions by adding scaled noise / bias.

    ``intensity`` controls the magnitude of the shift as a fraction of each
    column's standard deviation (e.g. 0.3 → shift mean by 0.3 * std).
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    cols = affected_columns or [c for c in _DEFAULT_NUMERIC_COLS if c in df.columns]

    n = len(df)
    for col in cols:
        if col not in df.columns:
            logger.warning("Covariate drift: column '%s' not found, skipping.", col)
            continue

        series = pd.to_numeric(df[col], errors="coerce")
        std = series.std(skipna=True)
        if pd.isna(std) or std == 0:
            std = 1.0

        shift = intensity * std

        if gradual:
            # Drift increases linearly from 0 → shift over the batch
            progressive_shift = np.linspace(0, shift, n)
            noise = rng.normal(loc=0, scale=std * 0.05, size=n)
            df[col] = series + progressive_shift + noise
        else:
            # Sudden shift applied to a random subset of rows
            n_affected = max(1, int(n * intensity))
            idx = rng.choice(n, size=n_affected, replace=False)
            noise = rng.normal(loc=shift, scale=std * 0.1, size=n_affected)
            df.loc[df.index[idx], col] = series.iloc[idx].values + noise

        logger.debug("Covariate drift injected into '%s' | shift=%.4f | gradual=%s", col, shift, gradual)

    return df


# ---------------------------------------------------------------------------
# 2. Label drift
# ---------------------------------------------------------------------------

def inject_label_drift(
    df: pd.DataFrame,
    *,
    intensity: float = 0.3,
    target_positive_rate: Optional[float] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Flip a fraction of labels to simulate class distribution shift.

    ``target_positive_rate`` overrides ``intensity`` when provided.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()

    if _LABEL_COL not in df.columns:
        logger.warning("Label drift: '%s' column not found. Skipping.", _LABEL_COL)
        return df

    labels = df[_LABEL_COL].copy()
    n = len(labels)

    if target_positive_rate is not None:
        # Force the positive rate to the target value
        current_pos = labels.sum()
        desired_pos = int(target_positive_rate * n)
        delta = desired_pos - current_pos

        if delta > 0:
            # Need more positives — flip negatives → positive
            neg_idx = labels[labels == 0].index.tolist()
            flip_idx = rng.choice(neg_idx, size=min(delta, len(neg_idx)), replace=False)
            labels.loc[flip_idx] = 1
        elif delta < 0:
            # Need fewer positives — flip positives → negative
            pos_idx = labels[labels == 1].index.tolist()
            flip_idx = rng.choice(pos_idx, size=min(-delta, len(pos_idx)), replace=False)
            labels.loc[flip_idx] = 0
    else:
        # Randomly flip ``intensity`` fraction of labels
        n_flip = max(1, int(n * intensity))
        flip_idx = rng.choice(n, size=n_flip, replace=False)
        labels.iloc[flip_idx] = 1 - labels.iloc[flip_idx]

    df[_LABEL_COL] = labels
    new_rate = df[_LABEL_COL].mean()
    logger.debug("Label drift injected. New positive rate=%.4f", new_rate)
    return df


# ---------------------------------------------------------------------------
# 3. Concept drift
# ---------------------------------------------------------------------------

def inject_concept_drift(
    df: pd.DataFrame,
    *,
    intensity: float = 0.3,
    gradual: bool = False,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Break the feature→label relationship by re-assigning labels for a subset
    of rows based on a *different* decision boundary.

    Simulates the real-world scenario where the same feature values now mean
    something different (e.g. fraud patterns evolve).
    """
    rng = np.random.default_rng(seed)
    df = df.copy()

    if _LABEL_COL not in df.columns:
        logger.warning("Concept drift: '%s' not found. Skipping.", _LABEL_COL)
        return df

    n = len(df)

    if gradual:
        # Probability of label flip increases linearly across the batch
        flip_prob = np.linspace(0, intensity, n)
        flip_mask = rng.random(n) < flip_prob
    else:
        n_affected = max(1, int(n * intensity))
        flip_mask = np.zeros(n, dtype=bool)
        flip_mask[rng.choice(n, size=n_affected, replace=False)] = True

    df.loc[flip_mask, _LABEL_COL] = 1 - df.loc[flip_mask, _LABEL_COL]

    # Also add mild covariate noise to the same rows to make it realistic
    for col in [c for c in _DEFAULT_NUMERIC_COLS if c in df.columns]:
        series = pd.to_numeric(df[col], errors="coerce")
        std = series.std(skipna=True) or 1.0
        noise = rng.normal(0, std * 0.2 * intensity, size=flip_mask.sum())
        df.loc[flip_mask, col] = series[flip_mask].values + noise

    logger.debug("Concept drift injected. Rows affected: %d / %d", flip_mask.sum(), n)
    return df


# ---------------------------------------------------------------------------
# 4. Missing value drift
# ---------------------------------------------------------------------------

def inject_missing_value_drift(
    df: pd.DataFrame,
    *,
    intensity: float = 0.3,
    affected_columns: Optional[list[str]] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Introduce NaN values into specified columns.
    ``intensity`` is the fraction of rows that will have nulls introduced.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    cols = affected_columns or [c for c in _DEFAULT_NUMERIC_COLS if c in df.columns]
    n = len(df)

    for col in cols:
        if col not in df.columns:
            continue
        n_null = max(1, int(n * intensity))
        null_idx = rng.choice(n, size=n_null, replace=False)
        df.iloc[null_idx, df.columns.get_loc(col)] = np.nan
        logger.debug("Missing value drift: %d nulls injected into '%s'.", n_null, col)

    return df


# ---------------------------------------------------------------------------
# 5. Schema drift
# ---------------------------------------------------------------------------

def inject_schema_drift(
    df: pd.DataFrame,
    *,
    drop_columns: Optional[list[str]] = None,
    add_columns: Optional[dict[str, Any]] = None,
    rename_columns: Optional[dict[str, str]] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Simulate schema changes: drop columns, add new ones, rename existing ones.

    Parameters
    ----------
    drop_columns:
        Column names to remove entirely.
    add_columns:
        ``{new_col_name: fill_value}`` — new columns with constant or callable fill.
    rename_columns:
        ``{old_name: new_name}`` mapping.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()

    if drop_columns:
        existing = [c for c in drop_columns if c in df.columns]
        df = df.drop(columns=existing)
        logger.debug("Schema drift: dropped columns %s.", existing)

    if add_columns:
        for col, fill in add_columns.items():
            if callable(fill):
                df[col] = fill(len(df), rng)
            else:
                df[col] = fill
            logger.debug("Schema drift: added column '%s'.", col)

    if rename_columns:
        existing_renames = {k: v for k, v in rename_columns.items() if k in df.columns}
        df = df.rename(columns=existing_renames)
        logger.debug("Schema drift: renamed columns %s.", existing_renames)

    return df


# ---------------------------------------------------------------------------
# 6. Categorical drift
# ---------------------------------------------------------------------------

def inject_categorical_drift(
    df: pd.DataFrame,
    *,
    intensity: float = 0.3,
    affected_columns: Optional[list[str]] = None,
    new_categories: Optional[dict[str, list[str]]] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Introduce unseen category labels into categorical columns.

    ``new_categories`` maps column names to lists of new (unseen) labels.
    Defaults to generic ``"unknown_<col>_<i>"`` labels if not provided.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    cols = affected_columns or [c for c in _DEFAULT_CAT_COLS if c in df.columns]
    n = len(df)

    for col in cols:
        if col not in df.columns:
            continue

        # Determine new labels to inject
        new_labels = (new_categories or {}).get(col, [f"unknown_{col}_0", f"unknown_{col}_1"])
        n_affected = max(1, int(n * intensity))
        idx = rng.choice(n, size=n_affected, replace=False)
        chosen_labels = rng.choice(new_labels, size=n_affected)
        df.iloc[idx, df.columns.get_loc(col)] = chosen_labels
        logger.debug(
            "Categorical drift: %d rows in '%s' set to unseen labels %s.",
            n_affected, col, new_labels,
        )

    return df


# ---------------------------------------------------------------------------
# 7. Temporal drift
# ---------------------------------------------------------------------------

def inject_temporal_drift(
    df: pd.DataFrame,
    *,
    gap_hours: float = 72.0,
    out_of_order_fraction: float = 0.1,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Introduce temporal anomalies:
    - A sudden gap (``gap_hours``) inserted after the midpoint of the batch.
    - A fraction of timestamps shuffled to simulate out-of-order events.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()

    if _TIMESTAMP_COL not in df.columns:
        logger.warning("Temporal drift: '%s' not found. Skipping.", _TIMESTAMP_COL)
        return df

    ts = pd.to_datetime(df[_TIMESTAMP_COL], utc=True, errors="coerce")
    n = len(ts)
    midpoint = n // 2

    # Insert time gap after midpoint
    gap = pd.Timedelta(hours=gap_hours)
    ts.iloc[midpoint:] = ts.iloc[midpoint:] + gap
    logger.debug("Temporal drift: gap of %.1fh inserted at row %d.", gap_hours, midpoint)

    # Shuffle a fraction of timestamps (out-of-order)
    n_shuffle = max(1, int(n * out_of_order_fraction))
    idx = rng.choice(n, size=n_shuffle, replace=False)
    shuffled_vals = ts.iloc[idx].values.copy()
    rng.shuffle(shuffled_vals)
    ts.iloc[idx] = shuffled_vals
    logger.debug("Temporal drift: %d timestamps shuffled (out-of-order).", n_shuffle)

    df[_TIMESTAMP_COL] = ts
    return df


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def inject_drift(df: pd.DataFrame, config: DriftConfig) -> pd.DataFrame:
    """
    Central dispatcher — routes to the correct injector based on ``config.drift_type``.

    Parameters
    ----------
    df:
        Input DataFrame (will not be mutated).
    config:
        ``DriftConfig`` describing what drift to apply.

    Returns
    -------
    pd.DataFrame
        New DataFrame with drift injected.
    """
    logger.info(
        "Injecting drift | type=%s | intensity=%.2f | gradual=%s | seed=%d",
        config.drift_type, config.intensity, config.gradual, config.seed,
    )

    kwargs: dict[str, Any] = dict(
        intensity=config.intensity,
        seed=config.seed,
        **config.extra,
    )

    match config.drift_type:
        case DriftType.NONE:
            return df.copy()

        case DriftType.COVARIATE:
            return inject_covariate_drift(
                df,
                affected_columns=config.affected_columns or None,
                gradual=config.gradual,
                **kwargs,
            )

        case DriftType.LABEL:
            return inject_label_drift(df, **kwargs)

        case DriftType.CONCEPT:
            return inject_concept_drift(df, gradual=config.gradual, **kwargs)

        case DriftType.MISSING_VALUE:
            return inject_missing_value_drift(
                df,
                affected_columns=config.affected_columns or None,
                **kwargs,
            )

        case DriftType.SCHEMA:
            return inject_schema_drift(
                df,
                drop_columns=config.extra.get("drop_columns"),
                add_columns=config.extra.get("add_columns"),
                rename_columns=config.extra.get("rename_columns"),
                seed=config.seed,
            )

        case DriftType.CATEGORICAL:
            return inject_categorical_drift(
                df,
                affected_columns=config.affected_columns or None,
                new_categories=config.extra.get("new_categories"),
                **{k: v for k, v in kwargs.items() if k != "new_categories"},
            )

        case DriftType.TEMPORAL:
            return inject_temporal_drift(
                df,
                gap_hours=config.extra.get("gap_hours", 72.0),
                out_of_order_fraction=config.extra.get("out_of_order_fraction", 0.1),
                seed=config.seed,
            )

        case _:
            raise ValueError(f"Unknown drift type: '{config.drift_type}'")


# ---------------------------------------------------------------------------
# Composite injector  (used by Airflow DAGs to chain multiple drift types)
# ---------------------------------------------------------------------------

def inject_composite_drift(
    df: pd.DataFrame,
    configs: list[DriftConfig],
) -> pd.DataFrame:
    """
    Apply multiple drift injections sequentially.

    Parameters
    ----------
    df:
        Input DataFrame.
    configs:
        Ordered list of ``DriftConfig`` objects. Applied left-to-right.

    Returns
    -------
    pd.DataFrame
        DataFrame with all drift types applied in order.
    """
    for cfg in configs:
        df = inject_drift(df, cfg)
    return df
# """
# data/validate.py
# ----------------
# Validates raw ingested DataFrames before feature engineering.

# Checks performed
# ----------------
# * Schema conformance  – column names & dtypes match the contract in schemas.py
# * Nullability         – non-nullable fields must have no missing values
# * Value ranges        – numeric columns stay within defined [min, max] bounds
# * Categorical levels  – categorical columns only contain known labels
# * Duplicate rows      – warns when duplicate primary-key rows are detected
# * Row-count guard     – raises if the batch is suspiciously small

# All failures are collected and returned as a structured ValidationReport so
# callers can decide whether to hard-fail or log-and-continue.
# """

# from __future__ import annotations

# import logging
# from dataclasses import dataclass, field
# from typing import Any

# import pandas as pd

# # from api.schemas import FEATURE_SCHEMA, FeatureSchema  # noqa: E402
# from api.schemas import PredictionInput, PredictionOutput  # for type hints and contract validation

# logger = logging.getLogger(__name__)

# # ---------------------------------------------------------------------------
# # Configuration / contracts (kept close to the code that uses them)
# # ---------------------------------------------------------------------------

# # Columns that must never be null
# NON_NULLABLE: list[str] = [
#     "customer_id",
#     "transaction_id",
#     "amount",
#     "timestamp",
# ]

# # Allowed numeric ranges  {column: (min_inclusive, max_inclusive)}
# NUMERIC_BOUNDS: dict[str, tuple[float, float]] = {
#     "amount": (0.0, 1_000_000.0),
#     "age": (0.0, 120.0),
#     "credit_score": (300.0, 850.0),
#     "num_transactions_30d": (0.0, 10_000.0),
# }

# # Allowed categories  {column: set_of_valid_labels}
# CATEGORICAL_LEVELS: dict[str, set[str]] = {
#     "currency": {"USD", "EUR", "GBP", "INR", "JPY"},
#     "channel": {"web", "mobile", "branch", "atm"},
#     "account_type": {"savings", "current", "credit"},
# }

# # Hard minimum number of rows per batch
# MIN_BATCH_ROWS: int = 10

# # Primary-key column (used for duplicate detection)
# PRIMARY_KEY: list[str] = ["transaction_id"]


# # ---------------------------------------------------------------------------
# # Report dataclass
# # ---------------------------------------------------------------------------

# @dataclass
# class ValidationError:
#     column: str
#     check: str
#     detail: str


# @dataclass
# class ValidationReport:
#     passed: bool = True
#     errors: list[ValidationError] = field(default_factory=list)
#     warnings: list[str] = field(default_factory=list)
#     row_count: int = 0
#     invalid_row_count: int = 0

#     def add_error(self, column: str, check: str, detail: str) -> None:
#         self.errors.append(ValidationError(column=column, check=check, detail=detail))
#         self.passed = False

#     def add_warning(self, msg: str) -> None:
#         self.warnings.append(msg)
#         logger.warning("Validation warning: %s", msg)

#     def summary(self) -> str:
#         status = "PASSED" if self.passed else "FAILED"
#         return (
#             f"Validation {status} | rows={self.row_count} "
#             f"| errors={len(self.errors)} | warnings={len(self.warnings)}"
#         )


# # ---------------------------------------------------------------------------
# # Individual check functions
# # ---------------------------------------------------------------------------

# def _check_row_count(df: pd.DataFrame, report: ValidationReport) -> None:
#     """Fail fast if the batch is too small to be meaningful."""
#     report.row_count = len(df)
#     if len(df) < MIN_BATCH_ROWS:
#         report.add_error(
#             column="*",
#             check="row_count",
#             detail=f"Batch has only {len(df)} rows; minimum is {MIN_BATCH_ROWS}.",
#         )


# # def _check_schema(df: pd.DataFrame, report: ValidationReport) -> None:
# #     """Verify required columns exist and have compatible dtypes."""
# #     required_columns: dict[str, type] = {
# #         # f.name: f.dtype for f in FEATURE_SCHEMA  # type: ignore[attr-defined]
# #         expected_columns = set(PredictionInput.model_fields.keys())
# #     }


#     missing = [col for col in required_columns if col not in df.columns]
#     if missing:
#         report.add_error(
#             column=str(missing),
#             check="schema_columns",
#             detail=f"Missing required columns: {missing}",
#         )

#     for col, expected_dtype in required_columns.items():
#         if col not in df.columns:
#             continue  # already flagged above
#         actual_kind = df[col].dtype.kind  # e.g. 'f', 'i', 'O', 'M'
#         if not _dtype_compatible(actual_kind, expected_dtype):
#             report.add_error(
#                 column=col,
#                 check="schema_dtype",
#                 detail=(
#                     f"Expected dtype category '{expected_dtype.__name__}', "
#                     f"got pandas dtype '{df[col].dtype}'."
#                 ),
#             )


# def _dtype_compatible(kind: str, expected: type) -> bool:
#     """Loose dtype compatibility check."""
#     mapping: dict[type, set[str]] = {
#         float: {"f", "i"},      # accept int columns for float fields
#         int: {"i", "u"},
#         str: {"O", "S", "U"},
#         bool: {"b", "i"},
#     }
#     return kind in mapping.get(expected, set())


# def _check_nullability(df: pd.DataFrame, report: ValidationReport) -> None:
#     """Ensure non-nullable columns have no missing values."""
#     for col in NON_NULLABLE:
#         if col not in df.columns:
#             continue
#         null_count = int(df[col].isna().sum())
#         if null_count > 0:
#             report.add_error(
#                 column=col,
#                 check="nullability",
#                 detail=f"Found {null_count} null values in non-nullable column.",
#             )


# def _check_numeric_bounds(df: pd.DataFrame, report: ValidationReport) -> None:
#     """Flag rows where numeric values fall outside expected bounds."""
#     for col, (lo, hi) in NUMERIC_BOUNDS.items():
#         if col not in df.columns:
#             continue
#         series = pd.to_numeric(df[col], errors="coerce")
#         out_of_range = series[(series < lo) | (series > hi)]
#         if not out_of_range.empty:
#             report.invalid_row_count += len(out_of_range)
#             report.add_error(
#                 column=col,
#                 check="numeric_bounds",
#                 detail=(
#                     f"{len(out_of_range)} values outside [{lo}, {hi}]. "
#                     f"Sample indices: {out_of_range.index[:5].tolist()}"
#                 ),
#             )


# def _check_categorical_levels(df: pd.DataFrame, report: ValidationReport) -> None:
#     """Detect unseen category labels."""
#     for col, allowed in CATEGORICAL_LEVELS.items():
#         if col not in df.columns:
#             continue
#         actual_levels = set(df[col].dropna().unique())
#         unknown = actual_levels - allowed
#         if unknown:
#             report.add_warning(
#                 f"Column '{col}' contains unknown categories: {unknown}. "
#                 f"Allowed: {allowed}"
#             )


# def _check_duplicates(df: pd.DataFrame, report: ValidationReport) -> None:
#     """Warn on duplicate primary-key rows."""
#     pk_cols = [c for c in PRIMARY_KEY if c in df.columns]
#     if not pk_cols:
#         return
#     dup_count = int(df.duplicated(subset=pk_cols).sum())
#     if dup_count > 0:
#         report.add_warning(
#             f"Found {dup_count} duplicate rows on key {pk_cols}."
#         )


# # ---------------------------------------------------------------------------
# # Public API
# # ---------------------------------------------------------------------------

# def validate_dataframe(df: pd.DataFrame, *, raise_on_failure: bool = False) -> ValidationReport:
#     """
#     Run all validation checks on *df* and return a :class:`ValidationReport`.

#     Parameters
#     ----------
#     df:
#         The raw DataFrame to validate.
#     raise_on_failure:
#         If True, raises ``ValueError`` when any hard check fails.

#     Returns
#     -------
#     ValidationReport
#         Structured report with `.passed`, `.errors`, and `.warnings`.
#     """
#     report = ValidationReport()

#     _check_row_count(df, report)
#     _check_schema(df, report)
#     _check_nullability(df, report)
#     _check_numeric_bounds(df, report)
#     _check_categorical_levels(df, report)
#     _check_duplicates(df, report)

#     logger.info(report.summary())

#     if not report.passed and raise_on_failure:
#         error_details = "\n".join(
#             f"  [{e.column}] {e.check}: {e.detail}" for e in report.errors
#         )
#         raise ValueError(f"Data validation failed:\n{error_details}")

#     return report


# def validate_single_record(record: dict[str, Any]) -> ValidationReport:
#     """
#     Convenience wrapper to validate a single prediction-time record.

#     Converts the dict to a one-row DataFrame and runs all checks,
#     skipping the row-count guard (single records are always < MIN_BATCH_ROWS).
#     """
#     df = pd.DataFrame([record])
#     # Temporarily lower the threshold so single records don't fail the count check
#     original_min = globals()["MIN_BATCH_ROWS"]
#     globals()["MIN_BATCH_ROWS"] = 1
#     try:
#         report = validate(df, raise_on_failure=False)
#     finally:
#         globals()["MIN_BATCH_ROWS"] = original_min
#     return report
"""
data/validate.py
----------------
Validates raw ingested DataFrames before feature engineering.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from api.schemas import PredictionInput

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MIN_BATCH_ROWS: int = 10

PRIMARY_KEY: list[str] = ["transaction_id"]


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------


@dataclass
class ValidationError:
    column: str
    check: str
    detail: str


@dataclass
class ValidationReport:
    passed: bool = True
    errors: list[ValidationError] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    row_count: int = 0
    invalid_row_count: int = 0

    def add_error(self, column: str, check: str, detail: str) -> None:
        self.errors.append(ValidationError(column=column, check=check, detail=detail))
        self.passed = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)
        logger.warning("Validation warning: %s", msg)

    def summary(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        return (
            f"Validation {status} | rows={self.row_count} "
            f"| errors={len(self.errors)} | warnings={len(self.warnings)}"
        )


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def _check_row_count(df: pd.DataFrame, report: ValidationReport) -> None:
    report.row_count = len(df)
    if len(df) < MIN_BATCH_ROWS:
        report.add_error(
            column="*",
            check="row_count",
            detail=f"Batch has only {len(df)} rows; minimum is {MIN_BATCH_ROWS}.",
        )


def _check_schema(df: pd.DataFrame, report: ValidationReport) -> None:
    """Schema validation using PredictionInput"""
    expected_columns = set(PredictionInput.model_fields.keys())
    actual_columns = set(df.columns)

    missing = expected_columns - actual_columns
    if missing:
        report.add_error(
            column=str(missing),
            check="schema_columns",
            detail=f"Missing required columns: {missing}",
        )

    extra = actual_columns - expected_columns
    if extra:
        report.add_warning(f"Extra columns present (not part of schema): {extra}")


def _check_duplicates(df: pd.DataFrame, report: ValidationReport) -> None:
    pk_cols = [c for c in PRIMARY_KEY if c in df.columns]
    if not pk_cols:
        return
    dup_count = int(df.duplicated(subset=pk_cols).sum())
    if dup_count > 0:
        report.add_warning(f"Found {dup_count} duplicate rows on key {pk_cols}.")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_dataframe(
    df: pd.DataFrame, *, raise_on_failure: bool = False
) -> ValidationReport:
    report = ValidationReport()

    _check_row_count(df, report)
    _check_schema(df, report)
    _check_duplicates(df, report)

    logger.info(report.summary())

    if not report.passed and raise_on_failure:
        error_details = "\n".join(
            f"  [{e.column}] {e.check}: {e.detail}" for e in report.errors
        )
        raise ValueError(f"Data validation failed:\n{error_details}")

    return report


def validate_single_record(record: dict[str, Any]) -> ValidationReport:
    df = pd.DataFrame([record])

    original_min = globals()["MIN_BATCH_ROWS"]
    globals()["MIN_BATCH_ROWS"] = 1
    try:
        report = validate_dataframe(df, raise_on_failure=False)
    finally:
        globals()["MIN_BATCH_ROWS"] = original_min

    return report

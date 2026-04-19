"""
monitoring/quality_report.py
Runs data-quality checks on the live prediction window and produces a
QualityReport that mirrors the structure of data/validate.py's ValidationReport
but targets log records rather than raw ingestion batches.

Checks
──────
  1. Missing-value rate per feature          (hard:  > MISSING_CRITICAL_PCT)
  2. Out-of-range values (numerical)         (hard:  > OOR_CRITICAL_PCT of rows)
  3. Unknown categories (categorical)        (soft:  > UNKNOWN_CAT_PCT of rows)
  4. Duplicate request IDs                   (hard:  any duplicates)
  5. Confidence distribution sanity          (soft:  mean < CONFIDENCE_LOW_WARN)
  6. Schema / column presence                (hard:  expected columns present)
  7. Prediction-class coverage               (soft:  unseen classes appear)

Output
──────
  QualityReport dataclass  →  .to_dict() / .save()
  Saved to artifacts/quality_reports/quality_report_<ts>.json
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", "artifacts"))
QUALITY_REPORTS_DIR = ARTIFACTS_DIR / "quality_reports"
QUALITY_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

MISSING_WARN_PCT = float(os.getenv("MISSING_WARN_PCT", "0.05"))  # 5 %
MISSING_CRITICAL_PCT = float(os.getenv("MISSING_CRITICAL_PCT", "0.15"))  # 15 %
OOR_CRITICAL_PCT = float(os.getenv("OOR_CRITICAL_PCT", "0.10"))  # 10 %
UNKNOWN_CAT_PCT = float(os.getenv("UNKNOWN_CAT_PCT", "0.05"))  # 5 %
CONFIDENCE_LOW_WARN = float(os.getenv("CONFIDENCE_LOW_WARN", "0.55"))
DUPLICATE_ID_HARD = True  # any duplicate request_id → hard fail


# ── Data structures ───────────────────────────────────────────────────────────


@dataclass
class CheckResult:
    check_name: str
    severity: str  # "hard" | "soft"
    passed: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeatureQuality:
    feature: str
    missing_pct: float
    missing_count: int
    oor_pct: Optional[float]  # numerical only
    oor_count: Optional[int]
    unknown_cat_pct: Optional[float]  # categorical only
    unknown_cat_count: Optional[int]
    quality_ok: bool


@dataclass
class QualityReport:
    report_id: str
    generated_at: str
    model_version: str
    window_size: int
    window_start: Optional[str]
    window_end: Optional[str]
    overall_passed: bool
    hard_failures: List[str]
    soft_warnings: List[str]
    checks: List[CheckResult]
    feature_quality: List[FeatureQuality]
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: Optional[Path] = None) -> Path:
        if path is None:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            path = QUALITY_REPORTS_DIR / f"quality_report_{ts}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        log.info("Quality report saved → %s", path)
        return path


# ── Individual checks ─────────────────────────────────────────────────────────


def _check_schema(df: pd.DataFrame, expected_columns: List[str]) -> CheckResult:
    missing = [c for c in expected_columns if c not in df.columns]
    passed = len(missing) == 0
    return CheckResult(
        check_name="schema_presence",
        severity="hard",
        passed=passed,
        message=(
            "All expected columns present." if passed else f"Missing columns: {missing}"
        ),
        details={"missing_columns": missing},
    )


def _check_duplicates(df: pd.DataFrame, id_col: str = "request_id") -> CheckResult:
    if id_col not in df.columns:
        return CheckResult(
            "duplicate_ids", "hard", True, f"Column '{id_col}' not found; skipped."
        )
    dupes = int(df[id_col].duplicated().sum())
    passed = dupes == 0
    return CheckResult(
        check_name="duplicate_ids",
        severity="hard",
        passed=passed,
        message=(
            "No duplicate request IDs."
            if passed
            else f"{dupes} duplicate request IDs found."
        ),
        details={"duplicate_count": dupes},
    )


def _check_confidence(df: pd.DataFrame) -> CheckResult:
    if "confidence" not in df.columns:
        return CheckResult(
            "confidence_sanity", "soft", True, "No confidence column; skipped."
        )
    mean_conf = float(df["confidence"].mean())
    passed = mean_conf >= CONFIDENCE_LOW_WARN
    return CheckResult(
        check_name="confidence_sanity",
        severity="soft",
        passed=passed,
        message=(
            f"Mean confidence {mean_conf:.3f} OK."
            if passed
            else f"Low mean confidence {mean_conf:.3f} < threshold {CONFIDENCE_LOW_WARN}."
        ),
        details={
            "mean_confidence": round(mean_conf, 4),
            "threshold": CONFIDENCE_LOW_WARN,
        },
    )


def _check_prediction_coverage(
    df: pd.DataFrame,
    known_classes: Optional[List[str]],
) -> CheckResult:
    if "predicted_class" not in df.columns or not known_classes:
        return CheckResult(
            "prediction_coverage", "soft", True, "Skipped — no class list provided."
        )
    seen = set(df["predicted_class"].unique())
    unknown = seen - set(known_classes)
    passed = len(unknown) == 0
    return CheckResult(
        check_name="prediction_coverage",
        severity="soft",
        passed=passed,
        message=(
            "All predicted classes known."
            if passed
            else f"Unknown classes in predictions: {unknown}"
        ),
        details={"unknown_classes": list(unknown), "known_classes": known_classes},
    )


# ── Feature-level quality ─────────────────────────────────────────────────────


def _feature_quality(
    df: pd.DataFrame,
    feature: str,
    ref_stats: Optional[Dict[str, Any]] = None,  # {"min": float, "max": float}
    known_categories: Optional[List[str]] = None,
) -> FeatureQuality:
    col = df[feature]
    n = len(col)
    missing = int(col.isna().sum())
    missing_pct = round(missing / n, 4) if n > 0 else 0.0

    oor_count, oor_pct = None, None
    unknown_cat_count, unknown_cat_pct = None, None

    if pd.api.types.is_numeric_dtype(col) and ref_stats:
        lo, hi = ref_stats.get("min"), ref_stats.get("max")
        if lo is not None and hi is not None:
            oor = int(((col < lo) | (col > hi)).sum())
            oor_count = oor
            oor_pct = round(oor / n, 4) if n > 0 else 0.0

    elif known_categories is not None:
        unknown = int((~col.isin(known_categories)).sum())
        unknown_cat_count = unknown
        unknown_cat_pct = round(unknown / n, 4) if n > 0 else 0.0

    quality_ok = (
        missing_pct < MISSING_CRITICAL_PCT
        and (oor_pct is None or oor_pct < OOR_CRITICAL_PCT)
        and (unknown_cat_pct is None or unknown_cat_pct < UNKNOWN_CAT_PCT)
    )

    return FeatureQuality(
        feature=feature,
        missing_pct=missing_pct,
        missing_count=missing,
        oor_pct=oor_pct,
        oor_count=oor_count,
        unknown_cat_pct=unknown_cat_pct,
        unknown_cat_count=unknown_cat_count,
        quality_ok=quality_ok,
    )


def _missing_rate_checks(
    df: pd.DataFrame, feature_cols: List[str]
) -> List[CheckResult]:
    results = []
    for feat in feature_cols:
        if feat not in df.columns:
            continue
        rate = float(df[feat].isna().mean())
        if rate >= MISSING_CRITICAL_PCT:
            results.append(
                CheckResult(
                    check_name=f"missing_rate_{feat}",
                    severity="hard",
                    passed=False,
                    message=f"Feature '{feat}' missing rate {rate:.1%} ≥ critical threshold {MISSING_CRITICAL_PCT:.1%}.",
                    details={"feature": feat, "missing_pct": round(rate, 4)},
                )
            )
        elif rate >= MISSING_WARN_PCT:
            results.append(
                CheckResult(
                    check_name=f"missing_rate_{feat}",
                    severity="soft",
                    passed=False,
                    message=f"Feature '{feat}' missing rate {rate:.1%} ≥ warning threshold {MISSING_WARN_PCT:.1%}.",
                    details={"feature": feat, "missing_pct": round(rate, 4)},
                )
            )
    return results


# ── Main entry point ──────────────────────────────────────────────────────────


def compute_quality_report(
    log_df: pd.DataFrame,
    feature_columns: Optional[List[str]] = None,
    expected_columns: Optional[List[str]] = None,
    ref_feature_stats: Optional[Dict[str, Dict[str, Any]]] = None,
    known_categories: Optional[Dict[str, List[str]]] = None,
    known_classes: Optional[List[str]] = None,
    model_version: str = "unknown",
    window_start: Optional[str] = None,
    window_end: Optional[str] = None,
    save: bool = True,
) -> QualityReport:
    """
    Parameters
    ──────────
    log_df             : DataFrame of prediction log records (from logger.query_logs)
    feature_columns    : Feature cols to inspect (auto-detected if None)
    expected_columns   : Hard schema check list
    ref_feature_stats  : {"feat": {"min": v, "max": v}} for OOR checks
    known_categories   : {"feat": ["cat1", "cat2"]} for categorical unknown checks
    known_classes      : Expected predicted_class values
    save               : Persist JSON to QUALITY_REPORTS_DIR
    """
    ts = datetime.now(timezone.utc)
    report_id = f"quality_{ts.strftime('%Y%m%d_%H%M%S')}"

    if feature_columns is None:
        exclude = {
            "request_id",
            "predicted_class",
            "confidence",
            "model_version",
            "timestamp",
            "label",
            "features",
        }
        feature_columns = [c for c in log_df.columns if c not in exclude]

    checks: List[CheckResult] = []

    # Schema check
    if expected_columns:
        checks.append(_check_schema(log_df, expected_columns))

    # Duplicate IDs
    checks.append(_check_duplicates(log_df))

    # Per-feature missing rates (generates per-feature CheckResults for failures only)
    checks.extend(_missing_rate_checks(log_df, feature_columns))

    # Confidence sanity
    checks.append(_check_confidence(log_df))

    # Prediction class coverage
    checks.append(_check_prediction_coverage(log_df, known_classes))

    # Per-feature quality objects
    feature_quality: List[FeatureQuality] = []
    for feat in feature_columns:
        if feat not in log_df.columns:
            continue
        ref_stats = (ref_feature_stats or {}).get(feat)
        known_cats = (known_categories or {}).get(feat)
        try:
            fq = _feature_quality(
                log_df, feat, ref_stats=ref_stats, known_categories=known_cats
            )
            feature_quality.append(fq)
        except Exception as e:
            log.error("FeatureQuality failed for '%s': %s", feat, e)

    hard_failures = [
        c.check_name for c in checks if not c.passed and c.severity == "hard"
    ]
    soft_warnings = [
        c.check_name for c in checks if not c.passed and c.severity == "soft"
    ]
    overall_passed = len(hard_failures) == 0

    summary = {
        "window_size": len(log_df),
        "features_checked": len(feature_quality),
        "hard_failures": len(hard_failures),
        "soft_warnings": len(soft_warnings),
        "features_with_issues": sum(1 for fq in feature_quality if not fq.quality_ok),
        "overall_passed": overall_passed,
    }

    report = QualityReport(
        report_id=report_id,
        generated_at=ts.isoformat(),
        model_version=model_version,
        window_size=len(log_df),
        window_start=window_start,
        window_end=window_end,
        overall_passed=overall_passed,
        hard_failures=hard_failures,
        soft_warnings=soft_warnings,
        checks=checks,
        feature_quality=feature_quality,
        summary=summary,
    )

    if save:
        report.save()

    log.info(
        "Quality report %s | passed=%s | hard=%d soft=%d",
        report_id,
        overall_passed,
        len(hard_failures),
        len(soft_warnings),
    )
    return report


def load_latest_quality_report() -> Optional[QualityReport]:
    reports = sorted(QUALITY_REPORTS_DIR.glob("quality_report_*.json"), reverse=True)
    if not reports:
        return None
    with open(reports[0]) as f:
        data = json.load(f)
    checks = [CheckResult(**c) for c in data.pop("checks", [])]
    fq = [FeatureQuality(**q) for q in data.pop("feature_quality", [])]
    return QualityReport(**data, checks=checks, feature_quality=fq)

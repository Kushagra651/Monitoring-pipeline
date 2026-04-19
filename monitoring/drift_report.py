"""
monitoring/drift_report.py
Computes feature + prediction drift between a reference dataset and
a live window retrieved from api/logger.py.

Drift methods
─────────────
  Numerical  : KS-test  (p-value + statistic)
  Categorical: Chi-square (p-value + statistic)
  PSI        : Population Stability Index (binned)
  Prediction : Distribution shift on predicted_class + confidence

Output
──────
  DriftReport  dataclass  (JSON-serialisable via .to_dict())
  Saved to  artifacts/drift_reports/drift_report_<timestamp>.json
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

log = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────────────────────

ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", "artifacts"))
DRIFT_REPORTS_DIR = ARTIFACTS_DIR / "drift_reports"
DRIFT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Thresholds
KS_P_THRESHOLD = float(os.getenv("KS_P_THRESHOLD", "0.05"))
CHI2_P_THRESHOLD = float(os.getenv("CHI2_P_THRESHOLD", "0.05"))
PSI_WARNING = float(os.getenv("PSI_WARNING", "0.1"))
PSI_CRITICAL = float(os.getenv("PSI_CRITICAL", "0.2"))
MIN_SAMPLES = int(os.getenv("DRIFT_MIN_SAMPLES", "30"))
PSI_BINS = int(os.getenv("PSI_BINS", "10"))


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class FeatureDriftResult:
    feature: str
    dtype: str                    # "numerical" | "categorical"
    method: str                   # "ks" | "chi2"
    statistic: float
    p_value: float
    psi: Optional[float]
    drifted: bool
    severity: str                 # "none" | "warning" | "critical"
    ref_mean: Optional[float] = None
    cur_mean: Optional[float] = None
    ref_std: Optional[float] = None
    cur_std: Optional[float] = None
    ref_top_values: Optional[Dict[str, float]] = None
    cur_top_values: Optional[Dict[str, float]] = None


@dataclass
class PredictionDriftResult:
    method: str = "chi2"
    statistic: float = 0.0
    p_value: float = 1.0
    drifted: bool = False
    psi: Optional[float] = None
    severity: str = "none"
    ref_class_dist: Dict[str, float] = field(default_factory=dict)
    cur_class_dist: Dict[str, float] = field(default_factory=dict)
    ref_confidence_mean: Optional[float] = None
    cur_confidence_mean: Optional[float] = None


@dataclass
class DriftReport:
    report_id: str
    generated_at: str
    model_version: str
    reference_size: int
    current_size: int
    window_start: Optional[str]
    window_end: Optional[str]
    overall_drifted: bool
    drifted_features: List[str]
    feature_results: List[FeatureDriftResult]
    prediction_drift: PredictionDriftResult
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: Optional[Path] = None) -> Path:
        if path is None:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            path = DRIFT_REPORTS_DIR / f"drift_report_{ts}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        log.info("Drift report saved → %s", path)
        return path


# ── PSI ──────────────────────────────────────────────────────────────────────

def _psi_numerical(ref: np.ndarray, cur: np.ndarray, bins: int = PSI_BINS) -> float:
    """Population Stability Index for a numerical feature."""
    eps = 1e-8
    # Build bins on ref, apply to cur
    breakpoints = np.percentile(ref, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 2:
        return 0.0

    ref_counts = np.histogram(ref, bins=breakpoints)[0]
    cur_counts = np.histogram(cur, bins=breakpoints)[0]

    ref_pct = ref_counts / (ref_counts.sum() + eps)
    cur_pct = cur_counts / (cur_counts.sum() + eps)

    ref_pct = np.where(ref_pct == 0, eps, ref_pct)
    cur_pct = np.where(cur_pct == 0, eps, cur_pct)

    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return round(psi, 6)


def _psi_categorical(ref: pd.Series, cur: pd.Series) -> float:
    eps = 1e-8
    categories = set(ref.unique()) | set(cur.unique())
    ref_pct = ref.value_counts(normalize=True).reindex(categories, fill_value=eps)
    cur_pct = cur.value_counts(normalize=True).reindex(categories, fill_value=eps)
    psi = float(np.sum((cur_pct - ref_pct) * np.log((cur_pct + eps) / (ref_pct + eps))))
    return round(psi, 6)


def _psi_severity(psi: float) -> str:
    if psi >= PSI_CRITICAL:
        return "critical"
    if psi >= PSI_WARNING:
        return "warning"
    return "none"


# ── Per-feature drift ─────────────────────────────────────────────────────────

def _drift_numerical(feature: str, ref: pd.Series, cur: pd.Series) -> FeatureDriftResult:
    r, c = ref.dropna().values, cur.dropna().values
    stat, p = stats.ks_2samp(r, c)
    psi = _psi_numerical(r, c)
    drifted = p < KS_P_THRESHOLD
    severity = "critical" if drifted and psi >= PSI_CRITICAL else \
               "warning"  if drifted or psi >= PSI_WARNING else "none"
    return FeatureDriftResult(
        feature=feature, dtype="numerical", method="ks",
        statistic=round(float(stat), 6), p_value=round(float(p), 6),
        psi=psi, drifted=drifted, severity=severity,
        ref_mean=round(float(ref.mean()), 4), cur_mean=round(float(cur.mean()), 4),
        ref_std=round(float(ref.std()), 4), cur_std=round(float(cur.std()), 4),
    )


def _drift_categorical(feature: str, ref: pd.Series, cur: pd.Series) -> FeatureDriftResult:
    categories = list(set(ref.unique()) | set(cur.unique()))
    ref_counts = ref.value_counts().reindex(categories, fill_value=0)
    cur_counts = cur.value_counts().reindex(categories, fill_value=0)

    # Chi-square needs at least 1 expected count per cell
    if ref_counts.sum() == 0 or cur_counts.sum() == 0:
        return FeatureDriftResult(
            feature=feature, dtype="categorical", method="chi2",
            statistic=0.0, p_value=1.0, psi=0.0,
            drifted=False, severity="none",
        )

    stat, p = stats.chisquare(
        f_obs=cur_counts.values + 1e-8,
        f_exp=ref_counts.values / ref_counts.sum() * cur_counts.sum() + 1e-8,
    )
    psi = _psi_categorical(ref, cur)
    drifted = p < CHI2_P_THRESHOLD

    top_n = 5
    ref_top = ref.value_counts(normalize=True).head(top_n).round(4).to_dict()
    cur_top = cur.value_counts(normalize=True).head(top_n).round(4).to_dict()

    severity = "critical" if drifted and psi >= PSI_CRITICAL else \
               "warning"  if drifted or psi >= PSI_WARNING else "none"
    return FeatureDriftResult(
        feature=feature, dtype="categorical", method="chi2",
        statistic=round(float(stat), 6), p_value=round(float(p), 6),
        psi=psi, drifted=drifted, severity=severity,
        ref_top_values=ref_top, cur_top_values=cur_top,
    )


# ── Prediction drift ──────────────────────────────────────────────────────────

def _drift_predictions(ref_logs: pd.DataFrame, cur_logs: pd.DataFrame) -> PredictionDriftResult:
    """Drift on predicted_class distribution + confidence mean shift."""
    result = PredictionDriftResult()

    if "predicted_class" not in ref_logs.columns or "predicted_class" not in cur_logs.columns:
        return result

    categories = list(set(ref_logs["predicted_class"].unique()) |
                      set(cur_logs["predicted_class"].unique()))
    ref_counts = ref_logs["predicted_class"].value_counts().reindex(categories, fill_value=0)
    cur_counts = cur_logs["predicted_class"].value_counts().reindex(categories, fill_value=0)

    stat, p = stats.chisquare(
        f_obs=cur_counts.values + 1e-8,
        f_exp=ref_counts.values / ref_counts.sum() * cur_counts.sum() + 1e-8,
    )
    psi = _psi_categorical(ref_logs["predicted_class"], cur_logs["predicted_class"])
    drifted = p < CHI2_P_THRESHOLD

    result.statistic = round(float(stat), 6)
    result.p_value = round(float(p), 6)
    result.drifted = drifted
    result.psi = psi
    result.severity = _psi_severity(psi) if not drifted else \
                      ("critical" if psi >= PSI_CRITICAL else "warning")
    result.ref_class_dist = ref_logs["predicted_class"].value_counts(normalize=True).round(4).to_dict()
    result.cur_class_dist = cur_logs["predicted_class"].value_counts(normalize=True).round(4).to_dict()

    if "confidence" in ref_logs.columns and "confidence" in cur_logs.columns:
        result.ref_confidence_mean = round(float(ref_logs["confidence"].mean()), 4)
        result.cur_confidence_mean = round(float(cur_logs["confidence"].mean()), 4)

    return result


# ── Main entry point ──────────────────────────────────────────────────────────

def compute_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    numerical_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
    model_version: str = "unknown",
    window_start: Optional[str] = None,
    window_end: Optional[str] = None,
    ref_logs: Optional[pd.DataFrame] = None,
    cur_logs: Optional[pd.DataFrame] = None,
    save: bool = True,
) -> DriftReport:
    """
    Parameters
    ──────────
    reference_df        : Training / baseline feature DataFrame
    current_df          : Live window feature DataFrame
    numerical_features  : Column names to treat as numerical (auto-detected if None)
    categorical_features: Column names to treat as categorical (auto-detected if None)
    ref_logs / cur_logs : Prediction log DataFrames for prediction drift
    save                : Persist report JSON to DRIFT_REPORTS_DIR
    """
    if len(current_df) < MIN_SAMPLES:
        raise ValueError(
            f"Current window has only {len(current_df)} rows; "
            f"need ≥ {MIN_SAMPLES} for reliable drift detection."
        )

    # Auto-detect feature types if not supplied
    common_cols = [c for c in reference_df.columns if c in current_df.columns]
    if numerical_features is None:
        numerical_features = [c for c in common_cols
                              if pd.api.types.is_numeric_dtype(reference_df[c])]
    if categorical_features is None:
        categorical_features = [c for c in common_cols
                                 if not pd.api.types.is_numeric_dtype(reference_df[c])]

    feature_results: List[FeatureDriftResult] = []

    for feat in numerical_features:
        if feat not in reference_df.columns or feat not in current_df.columns:
            log.warning("Numerical feature '%s' missing in one dataset, skipping.", feat)
            continue
        try:
            result = _drift_numerical(feat, reference_df[feat], current_df[feat])
            feature_results.append(result)
        except Exception as e:
            log.error("Error computing drift for '%s': %s", feat, e)

    for feat in categorical_features:
        if feat not in reference_df.columns or feat not in current_df.columns:
            log.warning("Categorical feature '%s' missing in one dataset, skipping.", feat)
            continue
        try:
            result = _drift_categorical(feat, reference_df[feat], current_df[feat])
            feature_results.append(result)
        except Exception as e:
            log.error("Error computing drift for '%s': %s", feat, e)

    # Prediction drift
    pred_drift = PredictionDriftResult()
    if ref_logs is not None and cur_logs is not None:
        try:
            pred_drift = _drift_predictions(ref_logs, cur_logs)
        except Exception as e:
            log.error("Prediction drift computation failed: %s", e)

    drifted_features = [r.feature for r in feature_results if r.drifted]
    overall_drifted = len(drifted_features) > 0 or pred_drift.drifted

    ts = datetime.now(timezone.utc)
    report_id = f"drift_{ts.strftime('%Y%m%d_%H%M%S')}"

    severities = [r.severity for r in feature_results] + [pred_drift.severity]
    summary = {
        "total_features_checked": len(feature_results),
        "drifted_count": len(drifted_features),
        "critical_count": severities.count("critical"),
        "warning_count": severities.count("warning"),
        "prediction_drifted": pred_drift.drifted,
        "drift_rate_pct": round(len(drifted_features) / max(len(feature_results), 1) * 100, 1),
    }

    report = DriftReport(
        report_id=report_id,
        generated_at=ts.isoformat(),
        model_version=model_version,
        reference_size=len(reference_df),
        current_size=len(current_df),
        window_start=window_start,
        window_end=window_end,
        overall_drifted=overall_drifted,
        drifted_features=drifted_features,
        feature_results=feature_results,
        prediction_drift=pred_drift,
        summary=summary,
    )

    if save:
        report.save()

    log.info(
        "Drift report %s | drifted=%s | features=%d/%d | prediction=%s",
        report_id, overall_drifted, len(drifted_features), len(feature_results), pred_drift.drifted,
    )
    return report


def load_latest_drift_report() -> Optional[DriftReport]:
    """Load most recent saved report (for DAG / exporter use)."""
    reports = sorted(DRIFT_REPORTS_DIR.glob("drift_report_*.json"), reverse=True)
    if not reports:
        return None
    with open(reports[0]) as f:
        data = json.load(f)
    feature_results = [FeatureDriftResult(**r) for r in data.pop("feature_results", [])]
    pred_drift = PredictionDriftResult(**data.pop("prediction_drift", {}))
    return DriftReport(**data, feature_results=feature_results, prediction_drift=pred_drift)
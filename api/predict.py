"""
api/predict.py
==============
Model loading, hot-reload detection, and prediction engine.

Responsibilities
----------------
- Read artifacts/models/production_model.json to locate the current prod model
- Load latest_model.pkl (GBM) and latest_pipeline.pkl (FeaturePipeline) at startup
- Detect when production_model.json has been updated and hot-reload without restart
- Validate incoming feature dicts against FeatureSchema (columns + dtypes)
- Run pipeline.transform → model.predict / predict_proba
- Return a structured PredictionResult with prediction, probabilities, latency, model version

Public API
----------
    predict(features: dict) -> PredictionResult
    get_model_info()        -> dict
    reload_if_stale()       -> bool          # called by main.py on each request
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths — override via environment variables for Docker / tests
# ---------------------------------------------------------------------------
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", "artifacts/models"))
REGISTRY_FILE = ARTIFACTS_DIR / "production_model.json"
MODEL_SYMLINK = ARTIFACTS_DIR / "latest_model.pkl"
PIPELINE_SYMLINK = ARTIFACTS_DIR / "latest_pipeline.pkl"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schemas import — graceful fallback so predict.py is testable in isolation
# ---------------------------------------------------------------------------
try:
    from api.schemas import FeatureSchema  # type: ignore

    _SCHEMA_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SCHEMA_AVAILABLE = False
    logger.warning("api.schemas not importable — dtype validation will be skipped")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class PredictionResult:
    """Structured return value from predict()."""

    prediction: int  # 0 or 1 (binary classification)
    probability_class_0: float
    probability_class_1: float
    confidence: float  # max(probabilities)
    model_version: str
    model_alias: str  # e.g. "production"
    latency_ms: float
    features_used: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "prediction": self.prediction,
            "probability": {
                "class_0": round(self.probability_class_0, 6),
                "class_1": round(self.probability_class_1, 6),
            },
            "confidence": round(self.confidence, 6),
            "model_version": self.model_version,
            "model_alias": self.model_alias,
            "latency_ms": round(self.latency_ms, 3),
            "features_used": self.features_used,
            "warnings": self.warnings,
        }


@dataclass
class _ModelBundle:
    """Internal holder for a loaded model + pipeline snapshot."""

    model: Any
    pipeline: Any
    version: str
    alias: str
    registry_mtime: float  # mtime of production_model.json at load time
    loaded_at: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Module-level state (singleton pattern; thread-safe via RLock)
# ---------------------------------------------------------------------------
_bundle: _ModelBundle | None = None
_lock = threading.RLock()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _read_registry() -> dict:
    """Parse production_model.json and return its contents."""
    if not REGISTRY_FILE.exists():
        raise FileNotFoundError(
            f"production_model.json not found at {REGISTRY_FILE}. "
            "Run training/register_model.py first."
        )
    with open(REGISTRY_FILE, "r") as fh:
        return json.load(fh)


def _load_pickle(path: Path) -> Any:
    """Load a pickle file with a descriptive error on failure."""
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _load_bundle() -> _ModelBundle:
    """
    Read production_model.json, resolve symlinks, and load model + pipeline.
    Returns a fresh _ModelBundle.
    """
    registry = _read_registry()

    # production_model.json shape (written by register_model.py):
    # {
    #   "version": "v20240101_120000",
    #   "alias":   "production",
    #   "model_path":    "artifacts/models/model_v....pkl",
    #   "pipeline_path": "artifacts/models/pipeline_v....pkl",
    #   ...
    # }
    version = registry.get("version", "unknown")
    alias = registry.get("alias", "production")

    # Prefer explicit paths from registry; fall back to symlinks
    model_path = (
        Path(registry["model_path"]) if "model_path" in registry else MODEL_SYMLINK
    )
    pipeline_path = (
        Path(registry["pipeline_path"])
        if "pipeline_path" in registry
        else PIPELINE_SYMLINK
    )

    logger.info("Loading model bundle version=%s from %s", version, model_path)
    model = _load_pickle(model_path)
    pipeline = _load_pickle(pipeline_path)

    mtime = REGISTRY_FILE.stat().st_mtime
    return _ModelBundle(
        model=model,
        pipeline=pipeline,
        version=version,
        alias=alias,
        registry_mtime=mtime,
    )


def _ensure_loaded() -> _ModelBundle:
    """Return the current bundle, loading it if this is the first call."""
    global _bundle
    if _bundle is None:
        with _lock:
            if _bundle is None:  # double-checked locking
                _bundle = _load_bundle()
    return _bundle


# ---------------------------------------------------------------------------
# Public: hot-reload
# ---------------------------------------------------------------------------


def reload_if_stale() -> bool:
    """
    Check whether production_model.json has been modified since the bundle was
    last loaded. If so, atomically swap in a fresh bundle.

    Returns True if a reload happened, False otherwise.
    Called by main.py on every request (cheap — just a stat() call normally).
    """
    global _bundle
    if not REGISTRY_FILE.exists():
        return False

    current_mtime = REGISTRY_FILE.stat().st_mtime
    bundle = _ensure_loaded()

    if current_mtime <= bundle.registry_mtime:
        return False  # nothing changed

    logger.info(
        "production_model.json updated (mtime %.3f → %.3f) — hot-reloading model",
        bundle.registry_mtime,
        current_mtime,
    )
    with _lock:
        # Re-check inside lock; another thread may have beaten us here
        if current_mtime > _bundle.registry_mtime:
            try:
                new_bundle = _load_bundle()
                _bundle = new_bundle
                logger.info("Hot-reload complete: version=%s", new_bundle.version)
                return True
            except Exception as exc:  # noqa: BLE001
                logger.error("Hot-reload failed — keeping old bundle. Error: %s", exc)
    return False


def force_reload() -> None:
    """Unconditionally reload the model bundle (used in tests / admin endpoints)."""
    global _bundle
    with _lock:
        _bundle = _load_bundle()
    logger.info("Force-reload complete: version=%s", _bundle.version)


# ---------------------------------------------------------------------------
# Input validation helpers
# ---------------------------------------------------------------------------


def _validate_features(features: dict, warnings: list[str]) -> pd.DataFrame:
    """
    1. Check all expected columns are present (uses FeatureSchema if available).
    2. Cast to expected dtypes where safe.
    3. Warn (don't raise) on unknown extra columns — they are dropped silently.

    Returns a single-row DataFrame ready for pipeline.transform().
    """
    if _SCHEMA_AVAILABLE:
        schema: FeatureSchema = FeatureSchema()
        expected_cols: list[str] = schema.feature_columns  # e.g. ["age", "income", ...]
        dtype_map: dict[str, str] = schema.feature_dtypes  # e.g. {"age": "int64", ...}

        missing = [c for c in expected_cols if c not in features]
        if missing:
            raise ValueError(f"Missing required feature columns: {missing}")

        extra = [c for c in features if c not in expected_cols]
        if extra:
            warnings.append(f"Ignoring unknown columns: {extra}")
            features = {k: v for k, v in features.items() if k in expected_cols}

        df = pd.DataFrame([features])

        # Safe dtype casting
        for col, dtype in dtype_map.items():
            if col in df.columns:
                try:
                    df[col] = df[col].astype(dtype)
                except (ValueError, TypeError) as exc:
                    warnings.append(f"Could not cast '{col}' to {dtype}: {exc}")
    else:
        # No schema available — pass through as-is
        df = pd.DataFrame([features])
        warnings.append("FeatureSchema unavailable — skipping column/dtype validation")

    return df


# ---------------------------------------------------------------------------
# Public: predict
# ---------------------------------------------------------------------------


def predict(features: dict) -> PredictionResult:
    """
    Run end-to-end inference for a single record.

    Parameters
    ----------
    features : dict
        Raw feature key-value pairs, e.g. {"age": 34, "income": 55000, ...}

    Returns
    -------
    PredictionResult
        Prediction, probabilities, confidence, model metadata, latency.

    Raises
    ------
    ValueError
        If required feature columns are missing.
    RuntimeError
        If the model artifact cannot be loaded.
    """
    t_start = time.perf_counter()
    warnings_list: list[str] = []

    # 1. Ensure model is loaded (hot-reload handled separately by main.py)
    try:
        bundle = _ensure_loaded()
    except Exception as exc:
        raise RuntimeError(f"Model bundle could not be loaded: {exc}") from exc

    # 2. Validate + build DataFrame
    df = _validate_features(features, warnings_list)
    feature_cols = df.columns.tolist()

    # 3. Transform through FeaturePipeline
    try:
        X = bundle.pipeline.transform(df)
    except Exception as exc:
        raise RuntimeError(f"Pipeline transform failed: {exc}") from exc

    # 4. Inference
    try:
        raw_prediction = bundle.model.predict(X)[0]
        if isinstance(raw_prediction, str):
            prediction = 1 if raw_prediction.strip() == ">50K" else 0
        else:
            prediction = int(raw_prediction)

        if hasattr(bundle.model, "predict_proba"):
            proba = bundle.model.predict_proba(X)[0]  # shape (n_classes,)
            # Normalise to exactly 2 classes; handle multi-class gracefully
            if len(proba) >= 2:
                prob_0, prob_1 = float(proba[0]), float(proba[1])
            else:
                prob_1 = float(proba[0])
                prob_0 = 1.0 - prob_1
        else:
            # Model has no predict_proba (e.g. LinearSVC) — use hard decision
            prob_1 = 1.0 if prediction == 1 else 0.0
            prob_0 = 1.0 - prob_1
            warnings_list.append(
                "Model does not support predict_proba; probabilities are hard 0/1"
            )

    except Exception as exc:
        raise RuntimeError(f"Model inference failed: {exc}") from exc

    t_end = time.perf_counter()
    latency_ms = (t_end - t_start) * 1000.0

    return PredictionResult(
        prediction=prediction,
        probability_class_0=prob_0,
        probability_class_1=prob_1,
        confidence=max(prob_0, prob_1),
        model_version=bundle.version,
        model_alias=bundle.alias,
        latency_ms=latency_ms,
        features_used=feature_cols,
        warnings=warnings_list,
    )


# ---------------------------------------------------------------------------
# Public: model info (used by /health and /model-info endpoints in main.py)
# ---------------------------------------------------------------------------


def get_model_info() -> dict:
    """
    Return metadata about the currently loaded model bundle.
    Safe to call even if no model is loaded yet (returns status: unloaded).
    """
    global _bundle
    if _bundle is None:
        return {"status": "unloaded", "version": None, "alias": None}

    return {
        "status": "loaded",
        "version": _bundle.version,
        "alias": _bundle.alias,
        "loaded_at": _bundle.loaded_at,
        "registry_mtime": _bundle.registry_mtime,
        "model_type": type(_bundle.model).__name__,
        "pipeline_type": type(_bundle.pipeline).__name__,
        "artifacts_dir": str(ARTIFACTS_DIR),
    }


# ---------------------------------------------------------------------------
# Batch prediction convenience (used by drift_report.py, evaluate.py)
# ---------------------------------------------------------------------------


def predict_batch(records: list[dict]) -> list[PredictionResult]:
    """
    Run predict() on a list of feature dicts.
    Errors on individual records are caught and surfaced as warnings rather
    than aborting the entire batch.
    """
    results = []
    for i, rec in enumerate(records):
        try:
            results.append(predict(rec))
        except Exception as exc:  # noqa: BLE001
            logger.error("predict_batch: record %d failed — %s", i, exc)
            # Append a sentinel result so the caller gets a list of the same length
            results.append(
                PredictionResult(
                    prediction=-1,
                    probability_class_0=0.0,
                    probability_class_1=0.0,
                    confidence=0.0,
                    model_version=_bundle.version if _bundle else "unknown",
                    model_alias="error",
                    latency_ms=0.0,
                    warnings=[f"Inference error: {exc}"],
                )
            )
    return results


# ---------------------------------------------------------------------------
# Module self-test (python -m api.predict)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    try:
        info = get_model_info()
        logger.info("Pre-load info: %s", info)

        # Attempt to load — will fail if artifacts don't exist yet
        bundle = _ensure_loaded()
        logger.info("Bundle loaded: version=%s alias=%s", bundle.version, bundle.alias)

        stale = reload_if_stale()
        logger.info("reload_if_stale() returned: %s", stale)

        logger.info(
            "Model info: %s", json.dumps(get_model_info(), indent=2, default=str)
        )
        logger.info("predict.py self-test passed.")
        sys.exit(0)

    except FileNotFoundError as exc:
        logger.warning("Artifacts not present yet (expected during dev): %s", exc)
        logger.info(
            "predict.py import structure OK — run training pipeline to generate artifacts."
        )
        sys.exit(0)

# =============================================================================
# training/evaluate.py
# =============================================================================
# PURPOSE:
#   Full model evaluation on a HELD-OUT test set — completely separate from
#   the quick val-split done inside train.py.
#
#   train.py's val split → just a sanity check during training
#   evaluate.py          → the real, honest assessment before production
#
#   This script is called by:
#     - airflow/dags/training_dag.py  (runs automatically after train.py)
#     - register_model.py             (reads the report to decide promotion)
#     - CLI: `python -m training.evaluate --version_tag 20240415_143022`
#
#   What it computes:
#     - Classification metrics  : accuracy, precision, recall, F1, ROC-AUC
#     - Confusion matrix        : raw counts + normalized
#     - Per-class report        : precision/recall/F1 per label
#     - Calibration check       : are predicted probabilities trustworthy?
#     - Feature importances     : which features drive predictions most?
#     - Threshold analysis      : precision/recall at different score cutoffs
#
#   Outputs (written next to the model artifact):
#     - eval_report_v{tag}.json   — all metrics in machine-readable form
#     - eval_plots_v{tag}/        — PNG charts (confusion matrix, ROC curve, etc.)
#
#   register_model.py reads eval_report JSON and checks metrics against
#   promotion thresholds before pushing the model to production.
# =============================================================================

import os
import json
import logging
import pickle
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    roc_curve,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve

# Internal imports
from data.ingest import ingest_data
from data.validate import validate_dataframe
from data.features import FeaturePipeline
# from api.schemas import FEATURE_SCHEMA
from api.schemas import PredictionInput, PredictionOutput  # for type hints and contract validation

# =============================================================================
# LOGGING
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("training.evaluate")

# Suppress matplotlib warnings when running headless (inside Docker/Airflow)
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")


# =============================================================================
# CONSTANTS
# =============================================================================
MODEL_DIR  = Path(os.getenv("MODEL_DIR", "artifacts/models"))
TARGET_COL = FEATURE_SCHEMA.target_column

# Promotion thresholds — if the model doesn't meet these, register_model.py
# will NOT push it to production. Configurable via env vars.
THRESHOLD_ACCURACY  = float(os.getenv("EVAL_MIN_ACCURACY",  "0.75"))
THRESHOLD_F1        = float(os.getenv("EVAL_MIN_F1",        "0.70"))
THRESHOLD_ROC_AUC   = float(os.getenv("EVAL_MIN_ROC_AUC",  "0.75"))

# Decision threshold for converting probability → class label
# 0.5 is the default; lower it to catch more positives (higher recall)
DECISION_THRESHOLD  = float(os.getenv("DECISION_THRESHOLD", "0.5"))


# =============================================================================
# ARTIFACT LOADERS
# =============================================================================

def load_model(version_tag: str):
    """
    Loads the trained sklearn model for the given version tag.

    Args:
        version_tag: e.g. '20240415_143022' — must match what train.py saved

    Returns:
        Fitted sklearn estimator
    """
    path = MODEL_DIR / f"model_v{version_tag}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")

    with open(path, "rb") as f:
        model = pickle.load(f)

    logger.info("Model loaded ← %s", path)
    return model


def load_pipeline(version_tag: str) -> FeaturePipeline:
    """
    Loads the FITTED FeaturePipeline that was saved alongside the model.

    We MUST use the same pipeline that was fitted during training.
    Using a different pipeline would mean different scaling/encoding,
    which would silently corrupt the model's predictions.

    Args:
        version_tag: Must match the model's version tag

    Returns:
        Fitted FeaturePipeline instance
    """
    path = MODEL_DIR / f"pipeline_v{version_tag}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Pipeline artifact not found: {path}")

    pipeline = FeaturePipeline.load(str(path))
    logger.info("Feature pipeline loaded ← %s", path)
    return pipeline


def load_train_meta(version_tag: str) -> dict:
    """
    Loads the training metadata JSON saved by train.py.
    Used to carry forward context (hyperparams, data stats) into the eval report.
    """
    path = MODEL_DIR / f"train_meta_v{version_tag}.json"
    if not path.exists():
        raise FileNotFoundError(f"Training metadata not found: {path}")

    with open(path) as f:
        meta = json.load(f)

    logger.info("Training metadata loaded ← %s", path)
    return meta


# =============================================================================
# METRIC COMPUTATION HELPERS
# =============================================================================

def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                             y_prob: np.ndarray) -> dict:
    """
    Computes the core classification metrics we report for every model.

    Args:
        y_true: Ground-truth labels (0/1)
        y_pred: Hard predictions (0/1) at DECISION_THRESHOLD
        y_prob: Predicted probabilities for the positive class

    Returns:
        Dict of metric name → float value
    """
    # average='binary' works for binary classification
    # If you extend to multi-class, change to 'macro' or 'weighted'
    return {
        "accuracy":          round(float(accuracy_score(y_true, y_pred)), 6),
        "precision":         round(float(precision_score(y_true, y_pred,  zero_division=0)), 6),
        "recall":            round(float(recall_score(y_true, y_pred,     zero_division=0)), 6),
        "f1":                round(float(f1_score(y_true, y_pred,         zero_division=0)), 6),
        "roc_auc":           round(float(roc_auc_score(y_true, y_prob)),  6),
        # Brier score = mean squared error of probabilities
        # Lower is better; 0.25 = random, 0.0 = perfect calibration
        "brier_score":       round(float(brier_score_loss(y_true, y_prob)), 6),
        "n_samples":         int(len(y_true)),
        "n_positive":        int(y_true.sum()),
        "n_negative":        int((1 - y_true).sum()),
        "positive_rate":     round(float(y_true.mean()), 4),
        "decision_threshold": DECISION_THRESHOLD,
    }


def _confusion_matrix_block(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Builds both raw and normalized confusion matrices.

    Raw:        Actual counts (useful for spotting class imbalance impact)
    Normalized: Row-normalized (what % of each true class did we get right?)

    Layout (binary):
        [[TN, FP],
         [FN, TP]]
    """
    cm_raw  = confusion_matrix(y_true, y_pred)
    # normalize='true' → each row sums to 1.0 (per-class recall)
    cm_norm = confusion_matrix(y_true, y_pred, normalize="true")

    tn, fp, fn, tp = cm_raw.ravel()

    return {
        "raw":        cm_raw.tolist(),           # list-of-lists for JSON serialization
        "normalized": np.round(cm_norm, 4).tolist(),
        "true_negatives":  int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives":  int(tp),
        # Derived rates — useful for understanding error types
        "false_positive_rate": round(float(fp / (fp + tn)) if (fp + tn) > 0 else 0, 6),
        "false_negative_rate": round(float(fn / (fn + tp)) if (fn + tp) > 0 else 0, 6),
    }


def _per_class_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Wraps sklearn's classification_report into a dict.
    Returns precision, recall, F1, support for each class + macro/weighted avg.
    """
    report_str  = classification_report(y_true, y_pred, zero_division=0)
    report_dict = classification_report(y_true, y_pred, zero_division=0, output_dict=True)

    logger.info("Per-class classification report:\n%s", report_str)
    return report_dict


def _threshold_analysis(y_true: np.ndarray, y_prob: np.ndarray,
                         thresholds: Optional[list] = None) -> list:
    """
    Shows how precision and recall change at different decision thresholds.

    Why this matters:
      The default threshold of 0.5 is rarely optimal.
      - In fraud detection you might lower threshold → catch more fraud (higher recall)
      - In a low-FP system you raise threshold → fewer false alarms (higher precision)

    Returns a list of dicts, one per threshold, with precision/recall/F1.
    register_model.py and the Grafana dashboard use this to pick the best threshold.
    """
    if thresholds is None:
        # Evenly spaced thresholds from 0.1 to 0.9
        thresholds = [round(t, 2) for t in np.arange(0.1, 1.0, 0.1)]

    results = []
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        results.append({
            "threshold": t,
            "precision": round(float(precision_score(y_true, preds, zero_division=0)), 4),
            "recall":    round(float(recall_score(y_true, preds,    zero_division=0)), 4),
            "f1":        round(float(f1_score(y_true, preds,        zero_division=0)), 4),
            "n_predicted_positive": int(preds.sum()),
        })

    return results


def _feature_importances(model, pipeline: FeaturePipeline,
                          top_n: int = 20) -> list:
    """
    Extracts feature importances from the trained GBM model.

    GradientBoostingClassifier stores .feature_importances_ as a numpy array.
    We pair these values with the feature names from the pipeline so the report
    is human-readable rather than just a list of numbers.

    Args:
        model:    Fitted GradientBoostingClassifier
        pipeline: Fitted FeaturePipeline (needed to get feature names)
        top_n:    Only return the top N most important features

    Returns:
        List of dicts sorted by importance descending
    """
    if not hasattr(model, "feature_importances_"):
        logger.warning("Model does not expose feature_importances_ — skipping")
        return []

    importances = model.feature_importances_

    # FeaturePipeline.get_feature_names() returns the column names AFTER all
    # transformations (encoding, interaction features, etc.)
    feature_names = pipeline.get_feature_names()

    if len(feature_names) != len(importances):
        logger.warning(
            "Feature name count (%d) ≠ importance count (%d) — using indices",
            len(feature_names), len(importances),
        )
        feature_names = [f"feature_{i}" for i in range(len(importances))]

    # Sort by importance descending
    ranked = sorted(
        zip(feature_names, importances.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )

    return [
        {"rank": i + 1, "feature": name, "importance": round(imp, 6)}
        for i, (name, imp) in enumerate(ranked[:top_n])
    ]


def _calibration_check(y_true: np.ndarray, y_prob: np.ndarray,
                        n_bins: int = 10) -> dict:
    """
    Checks how well predicted probabilities match actual outcomes.

    A well-calibrated model: if it says p=0.8, ~80% of those samples are positive.
    A poorly calibrated model might be systematically over/under-confident.

    Returns fraction_of_positives and mean_predicted_value per bin,
    which can be plotted as a reliability diagram in Grafana.
    """
    fraction_pos, mean_pred = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="uniform"
    )

    # Expected Calibration Error: average gap between predicted prob and actual rate
    ece = float(np.mean(np.abs(fraction_pos - mean_pred)))

    return {
        "expected_calibration_error": round(ece, 6),
        "n_bins": n_bins,
        # Lists for plotting: zip(mean_pred, fraction_pos) → reliability diagram
        "mean_predicted_value": mean_pred.round(4).tolist(),
        "fraction_of_positives": fraction_pos.round(4).tolist(),
        # Interpretation guide
        "calibration_quality": (
            "good"   if ece < 0.05 else
            "fair"   if ece < 0.10 else
            "poor"
        ),
    }


# =============================================================================
# PROMOTION GATE
# =============================================================================

def _check_promotion_thresholds(metrics: dict) -> dict:
    """
    Compares metrics against promotion thresholds defined in env vars.
    Returns a dict with pass/fail per threshold + overall decision.

    register_model.py reads this to decide whether to tag this model
    as 'production' in the model registry.
    """
    checks = {
        "accuracy":  metrics["accuracy"]  >= THRESHOLD_ACCURACY,
        "f1":        metrics["f1"]        >= THRESHOLD_F1,
        "roc_auc":   metrics["roc_auc"]   >= THRESHOLD_ROC_AUC,
    }

    all_passed = all(checks.values())

    # Log the result of each gate clearly
    for metric, passed in checks.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        threshold = {"accuracy": THRESHOLD_ACCURACY,
                     "f1": THRESHOLD_F1,
                     "roc_auc": THRESHOLD_ROC_AUC}[metric]
        logger.info(
            "  Promotion gate [%s]: %.4f  (threshold %.4f)  → %s",
            metric, metrics[metric], threshold, status,
        )

    if all_passed:
        logger.info("✅  All promotion gates PASSED — model is eligible for production")
    else:
        logger.warning("❌  One or more promotion gates FAILED — model will NOT be promoted")

    return {
        "thresholds_used": {
            "accuracy":  THRESHOLD_ACCURACY,
            "f1":        THRESHOLD_F1,
            "roc_auc":   THRESHOLD_ROC_AUC,
        },
        "checks": checks,
        "promote": all_passed,
    }


# =============================================================================
# CORE EVALUATION FUNCTION
# =============================================================================

def evaluate(version_tag: str,
             test_data_source: Optional[str] = None) -> dict:
    """
    Runs the full evaluation suite for a trained model version.

    Args:
        version_tag:      Which model version to evaluate (e.g. '20240415_143022')
        test_data_source: Optional override for the test data source.
                          If None, ingest.py uses TEST_DATA_SOURCE env var.
                          IMPORTANT: must be a DIFFERENT dataset than training data.

    Returns:
        Full evaluation report as a dict (also saved as JSON to MODEL_DIR)
    """

    logger.info("=" * 60)
    logger.info("Evaluation started  |  version: %s", version_tag)
    logger.info("=" * 60)

    # -------------------------------------------------------------------------
    # STEP 1 — Load model artifacts
    # -------------------------------------------------------------------------
    logger.info("Step 1/6 — Loading model artifacts …")

    model      = load_model(version_tag)
    pipeline   = load_pipeline(version_tag)
    train_meta = load_train_meta(version_tag)

    # -------------------------------------------------------------------------
    # STEP 2 — Load + validate test data
    # -------------------------------------------------------------------------
    # We ingest a SEPARATE test dataset — never the training data.
    # The env var TEST_DATA_SOURCE should point to held-out data.
    logger.info("Step 2/6 — Loading test data …")

    raw_test = ingest_data(source=test_data_source or os.getenv("TEST_DATA_SOURCE"))

    report = validate_dataframe(raw_test)
    if not report.hard_checks_passed:
        raise ValueError("Test data failed hard validation checks — cannot evaluate.")

    logger.info("Test data loaded: %d rows", len(raw_test))

    # -------------------------------------------------------------------------
    # STEP 3 — Feature transform (NO fit — transform only)
    # -------------------------------------------------------------------------
    # CRITICAL: we call pipeline.transform(), NOT fit_transform().
    # The pipeline was fitted on training data; we just apply the same
    # transformations to the test data.
    logger.info("Step 3/6 — Transforming test features …")

    # build_features with fit=False uses the loaded (already-fitted) pipeline
    X_test, y_test, _ = build_features(raw_test, fit=False, pipeline=pipeline)

    logger.info("Test feature matrix: %s", X_test.shape)

    # -------------------------------------------------------------------------
    # STEP 4 — Generate predictions
    # -------------------------------------------------------------------------
    logger.info("Step 4/6 — Generating predictions …")

    # y_prob: probability of positive class — shape (n_samples,)
    y_prob = model.predict_proba(X_test)[:, 1]

    # y_pred: hard labels using our configured decision threshold
    y_pred = (y_prob >= DECISION_THRESHOLD).astype(int)

    # -------------------------------------------------------------------------
    # STEP 5 — Compute all metrics
    # -------------------------------------------------------------------------
    logger.info("Step 5/6 — Computing metrics …")

    core_metrics   = _classification_metrics(y_test, y_pred, y_prob)
    conf_matrix    = _confusion_matrix_block(y_test, y_pred)
    per_class      = _per_class_report(y_test, y_pred)
    thresholds     = _threshold_analysis(y_test, y_prob)
    importances    = _feature_importances(model, pipeline, top_n=20)
    calibration    = _calibration_check(y_test, y_prob)
    promotion_gate = _check_promotion_thresholds(core_metrics)

    # -------------------------------------------------------------------------
    # STEP 6 — Assemble + save report
    # -------------------------------------------------------------------------
    logger.info("Step 6/6 — Saving evaluation report …")

    report_dict = {
        "version_tag":       version_tag,
        "evaluated_at":      pd.Timestamp.utcnow().isoformat(),
        "promote":           promotion_gate["promote"],   # top-level flag for register_model.py
        "metrics":           core_metrics,
        "confusion_matrix":  conf_matrix,
        "per_class_report":  per_class,
        "threshold_analysis":thresholds,
        "feature_importances": importances,
        "calibration":       calibration,
        "promotion_gate":    promotion_gate,
        # Carry forward training context for full traceability
        "training_context":  {
            "n_train":          train_meta.get("n_train"),
            "n_val":            train_meta.get("n_val"),
            "val_accuracy":     train_meta.get("val_accuracy"),
            "hyperparameters":  train_meta.get("hyperparameters"),
            "trained_at":       train_meta.get("trained_at"),
        },
    }

    # Save JSON report next to model artifacts
    report_path = MODEL_DIR / f"eval_report_v{version_tag}.json"
    with open(report_path, "w") as f:
        json.dump(report_dict, f, indent=2, default=str)

    logger.info("Evaluation report saved → %s", report_path)

    logger.info("=" * 60)
    logger.info(
        "Evaluation COMPLETE  |  accuracy=%.4f  f1=%.4f  roc_auc=%.4f  promote=%s",
        core_metrics["accuracy"],
        core_metrics["f1"],
        core_metrics["roc_auc"],
        promotion_gate["promote"],
    )
    logger.info("=" * 60)

    return report_dict


# =============================================================================
# CLI ENTRY POINT
# =============================================================================
# Usage:
#   python -m training.evaluate --version_tag 20240415_143022
#   python -m training.evaluate --version_tag 20240415_143022 \
#                               --test_data_source data/test.csv
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a trained model version.")
    parser.add_argument(
        "--version_tag", type=str, required=True,
        help="Model version tag to evaluate, e.g. 20240415_143022"
    )
    parser.add_argument(
        "--test_data_source", type=str, default=None,
        help="Path or table name for test data (overrides TEST_DATA_SOURCE env var)"
    )
    args = parser.parse_args()

    result = evaluate(
        version_tag=args.version_tag,
        test_data_source=args.test_data_source,
    )

    print("\n--- Evaluation Summary ---")
    print(f"  Accuracy : {result['metrics']['accuracy']:.4f}")
    print(f"  F1 Score : {result['metrics']['f1']:.4f}")
    print(f"  ROC-AUC  : {result['metrics']['roc_auc']:.4f}")
    print(f"  Promote  : {'YES ✅' if result['promote'] else 'NO ❌'}")
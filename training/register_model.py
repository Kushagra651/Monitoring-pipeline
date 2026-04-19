# =============================================================================
# training/register_model.py
# =============================================================================
# PURPOSE:
#   The final gate before a model goes to production.
#   This script reads the evaluation report produced by evaluate.py and
#   decides whether to "register" the model — i.e., tag it as the active
#   production model that predict.py will load and serve.
#
#   Think of this as the model's "deployment approval" step.
#
# WHAT IT DOES:
#   1. Reads eval_report_v{tag}.json  → checks the promote flag + metrics
#   2. Compares against the CURRENT production model (if one exists)
#      → only promotes if the new model is strictly better
#   3. Writes a registry file (model_registry.json) that acts as the
#      single source of truth for "which model is in production?"
#   4. Updates a `latest_production` symlink so predict.py always loads
#      the current model without being restarted
#   5. Archives the previous production model (keeps history, never deletes)
#   6. Writes a promotion audit log entry for traceability
#
# WHO CALLS THIS:
#   - airflow/dags/training_dag.py  (automatically, after evaluate.py passes)
#   - CLI: `python -m training.register_model --version_tag 20240415_143022`
#   - Can also be called manually to ROLLBACK: --action rollback --version_tag <old>
#
# OUTPUT FILES (all in MODEL_DIR):
#   - model_registry.json         → full registry of all model versions
#   - production_model.json       → just the current production model info
#                                   (predict.py reads this on startup)
#   - promotion_audit.log         → append-only human-readable audit trail
#   - latest_model.pkl  (symlink) → always points to current production model
#   - latest_pipeline.pkl (symlink)→ always points to current production pipeline
# =============================================================================

import os
import json
import logging
# import shutil
# import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# =============================================================================
# LOGGING
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("training.register_model")


# =============================================================================
# CONSTANTS
# =============================================================================
MODEL_DIR = Path(os.getenv("MODEL_DIR", "artifacts/models"))

# The registry file is the central database of all model versions.
# It's a JSON file — simple, human-readable, and diff-able in git.
REGISTRY_FILE        = MODEL_DIR / "model_registry.json"

# predict.py reads PRODUCTION_FILE on startup to know which model to load.
# It's a tiny JSON with just the version_tag + paths — fast to read.
PRODUCTION_FILE      = MODEL_DIR / "production_model.json"

# Append-only audit log — every promotion and rollback is recorded here.
AUDIT_LOG_FILE       = MODEL_DIR / "promotion_audit.log"

# Symlinks — predict.py uses these so it never needs to know the version_tag.
# When we promote a new model we just update the symlinks; no code changes.
LATEST_MODEL_SYMLINK    = MODEL_DIR / "latest_model.pkl"
LATEST_PIPELINE_SYMLINK = MODEL_DIR / "latest_pipeline.pkl"

# Minimum improvement a new model must show over current production.
# Prevents deploying a model that's only marginally better (could be noise).
MIN_IMPROVEMENT_DELTA = float(os.getenv("MIN_IMPROVEMENT_DELTA", "0.001"))  # 0.1%


# =============================================================================
# REGISTRY HELPERS
# =============================================================================

def _ensure_dirs() -> None:
    """Creates MODEL_DIR and any parent directories if they don't exist."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def _load_registry() -> dict:
    """
    Loads the model registry JSON from disk.

    The registry looks like:
    {
      "models": {
        "20240415_143022": {
          "version_tag": "...",
          "status": "production" | "archived" | "failed",
          "promoted_at": "...",
          "metrics": {...},
          ...
        },
        ...
      },
      "production_version": "20240415_143022"   ← or null if none yet
    }

    Returns an empty registry structure if the file doesn't exist yet
    (i.e., first ever training run).
    """
    if not REGISTRY_FILE.exists():
        logger.info("No existing registry found — starting fresh")
        return {"models": {}, "production_version": None}

    with open(REGISTRY_FILE) as f:
        registry = json.load(f)

    logger.info(
        "Registry loaded — %d versions tracked, production: %s",
        len(registry.get("models", {})),
        registry.get("production_version", "none"),
    )
    return registry


def _save_registry(registry: dict) -> None:
    """
    Persists the registry dict back to disk as pretty-printed JSON.
    Called after every promotion or status change.
    """
    with open(REGISTRY_FILE, "w") as f:
        json.dump(registry, f, indent=2, default=str)
    logger.info("Registry saved → %s", REGISTRY_FILE)


def _load_eval_report(version_tag: str) -> dict:
    """
    Loads the evaluation report produced by evaluate.py for a given version.

    Args:
        version_tag: e.g. '20240415_143022'

    Returns:
        The full evaluation report dict

    Raises:
        FileNotFoundError if evaluate.py hasn't been run for this version
    """
    path = MODEL_DIR / f"eval_report_v{version_tag}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Evaluation report not found: {path}\n"
            "Run evaluate.py first before registering a model."
        )

    with open(path) as f:
        report = json.load(f)

    logger.info("Eval report loaded ← %s", path)
    return report


def _load_production_model_info() -> Optional[dict]:
    """
    Loads the current production model info from production_model.json.

    Returns None if no model has been promoted to production yet.
    This is the expected state on the very first training run.
    """
    if not PRODUCTION_FILE.exists():
        logger.info("No production model currently registered")
        return None

    with open(PRODUCTION_FILE) as f:
        info = json.load(f)

    logger.info(
        "Current production model: version=%s  (promoted %s)",
        info.get("version_tag"),
        info.get("promoted_at"),
    )
    return info


# =============================================================================
# COMPARISON LOGIC
# =============================================================================

def _is_better_than_production(new_metrics: dict,
                                 prod_metrics: Optional[dict]) -> tuple[bool, str]:
    """
    Decides if the new model is good enough to replace the current production model.

    Decision rules (in order):
      1. If there is no production model → always promote (first deployment)
      2. New model must pass its own promotion gates (checked in evaluate.py)
      3. New model's ROC-AUC must be at least MIN_IMPROVEMENT_DELTA better
         than the current production model's ROC-AUC
         (primary metric — best single indicator of overall model quality)

    We use ROC-AUC as the comparison metric because:
      - It's threshold-independent (unlike accuracy or F1)
      - It captures the model's discrimination ability across all thresholds
      - It's robust to class imbalance

    Args:
        new_metrics:  Metrics dict from the new model's eval report
        prod_metrics: Metrics dict from the current production model (or None)

    Returns:
        (decision: bool, reason: str)
    """
    # Rule 1: No production model yet — just deploy
    if prod_metrics is None:
        return True, "No existing production model — first deployment"

    new_auc  = new_metrics.get("roc_auc", 0.0)
    prod_auc = prod_metrics.get("roc_auc", 0.0)
    delta    = new_auc - prod_auc

    new_f1   = new_metrics.get("f1", 0.0)
    prod_f1  = prod_metrics.get("f1", 0.0)

    logger.info(
        "Comparison — new ROC-AUC: %.4f  |  prod ROC-AUC: %.4f  |  delta: %+.4f  (min: %+.4f)",
        new_auc, prod_auc, delta, MIN_IMPROVEMENT_DELTA,
    )
    logger.info(
        "Comparison — new F1: %.4f  |  prod F1: %.4f",
        new_f1, prod_f1,
    )

    # Rule 3: Must beat production by at least MIN_IMPROVEMENT_DELTA
    if delta >= MIN_IMPROVEMENT_DELTA:
        return True, (
            f"New ROC-AUC {new_auc:.4f} exceeds production {prod_auc:.4f} "
            f"by {delta:+.4f} (min required: {MIN_IMPROVEMENT_DELTA:+.4f})"
        )
    elif delta >= 0:
        return False, (
            f"New ROC-AUC {new_auc:.4f} is better than production {prod_auc:.4f} "
            f"but improvement {delta:+.4f} is below minimum delta {MIN_IMPROVEMENT_DELTA:+.4f}. "
            "Not promoting to avoid deploying noise."
        )
    else:
        return False, (
            f"New ROC-AUC {new_auc:.4f} is WORSE than production {prod_auc:.4f} "
            f"(delta: {delta:+.4f}). Keeping current production model."
        )


# =============================================================================
# SYMLINK MANAGEMENT
# =============================================================================

def _update_symlinks(version_tag: str) -> None:
    """
    Updates the latest_model.pkl and latest_pipeline.pkl symlinks to point
    to the newly promoted model's artifacts.

    Why symlinks?
      predict.py always opens 'latest_model.pkl' — it never needs to know
      the version_tag. When we promote a new model, we update the symlink
      and predict.py picks it up on next load (or hot-reload) without any
      code change or container restart.

    Note: On Windows, symlinks require elevated permissions.
    In Docker (Linux) this works seamlessly.
    """
    model_path    = MODEL_DIR / f"model_v{version_tag}.pkl"
    pipeline_path = MODEL_DIR / f"pipeline_v{version_tag}.pkl"

    for symlink, target in [
        (LATEST_MODEL_SYMLINK,    model_path),
        (LATEST_PIPELINE_SYMLINK, pipeline_path),
    ]:
        # Remove old symlink if it exists
        if symlink.exists() or symlink.is_symlink():
            symlink.unlink()

        # Create new symlink pointing to versioned artifact
        symlink.symlink_to(target.name)   # relative symlink — portable across mounts
        logger.info("Symlink updated: %s → %s", symlink.name, target.name)


# =============================================================================
# AUDIT LOGGING
# =============================================================================

def _write_audit_entry(action: str, version_tag: str, reason: str,
                        metrics: Optional[dict] = None) -> None:
    """
    Appends a human-readable line to the audit log.

    The audit log is append-only — we never delete or overwrite entries.
    This gives a permanent record of every promotion and rollback, who
    triggered it (Airflow / CLI), and why.

    Format per line:
      2024-04-15T14:30:22Z | PROMOTED  | v20240415_143022 | roc_auc=0.8821 | reason: ...
    """
    timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    metrics_str = ""
    if metrics:
        metrics_str = (
            f"acc={metrics.get('accuracy', 'n/a'):.4f}  "
            f"f1={metrics.get('f1', 'n/a'):.4f}  "
            f"roc_auc={metrics.get('roc_auc', 'n/a'):.4f}"
        )

    entry = (
        f"{timestamp} | {action:<12} | v{version_tag} | "
        f"{metrics_str} | {reason}\n"
    )

    with open(AUDIT_LOG_FILE, "a") as f:
        f.write(entry)

    logger.info("Audit log entry written: %s", entry.strip())


# =============================================================================
# ARCHIVE PREVIOUS PRODUCTION
# =============================================================================

def _archive_current_production(registry: dict) -> None:
    """
    Marks the current production model as 'archived' in the registry.

    We NEVER delete old model artifacts — they're needed for:
      - Rollback (if the new model has a bug in production)
      - Audit / compliance (regulators may ask "what model was live on date X?")
      - Debugging (compare predictions between versions)

    This just updates the status field in the registry JSON.
    The .pkl files on disk are untouched.
    """
    current_prod = registry.get("production_version")
    if current_prod and current_prod in registry["models"]:
        registry["models"][current_prod]["status"] = "archived"
        registry["models"][current_prod]["archived_at"] = datetime.utcnow().isoformat()
        logger.info("Previous production model archived: v%s", current_prod)


# =============================================================================
# CORE REGISTRATION FUNCTION
# =============================================================================

def register_model(version_tag: str,
                   force: bool = False) -> dict:
    """
    Attempts to promote a trained + evaluated model to production.

    Full flow:
      1. Load the eval report for this version
      2. Check the promote flag set by evaluate.py
      3. Load current production model info for comparison
      4. Decide whether new model is better (or use force=True to skip)
      5. If promoting: archive old, update registry, update symlinks, write production_model.json
      6. Write audit log entry regardless of outcome
      7. Return a result dict (used by Airflow XCom)

    Args:
        version_tag: Version to attempt to promote (e.g. '20240415_143022')
        force:       If True, skip the comparison check and promote regardless.
                     USE WITH CAUTION — only for manual overrides / hotfixes.

    Returns:
        Dict with keys: promoted (bool), version_tag, reason, metrics
    """
    _ensure_dirs()

    logger.info("=" * 60)
    logger.info("Model registration started  |  version: %s  |  force=%s",
                version_tag, force)
    logger.info("=" * 60)

    # -------------------------------------------------------------------------
    # STEP 1 — Load evaluation report
    # -------------------------------------------------------------------------
    logger.info("Step 1/5 — Loading evaluation report …")
    eval_report = _load_eval_report(version_tag)
    new_metrics = eval_report["metrics"]

    # -------------------------------------------------------------------------
    # STEP 2 — Check evaluate.py's promotion gate
    # -------------------------------------------------------------------------
    # evaluate.py already ran all threshold checks and set promote=True/False.
    # We respect that decision here unless force=True.
    logger.info("Step 2/5 — Checking promotion gate from evaluate.py …")

    eval_promote = eval_report.get("promote", False)

    if not eval_promote and not force:
        reason = (
            "evaluate.py promotion gate FAILED "
            f"(accuracy={new_metrics.get('accuracy'):.4f}, "
            f"f1={new_metrics.get('f1'):.4f}, "
            f"roc_auc={new_metrics.get('roc_auc'):.4f}). "
            "Model does not meet minimum quality thresholds."
        )
        logger.warning("❌  Registration REJECTED: %s", reason)
        _write_audit_entry("REJECTED", version_tag, reason, new_metrics)
        _record_in_registry(version_tag, "failed", eval_report, reason)
        return {"promoted": False, "version_tag": version_tag,
                "reason": reason, "metrics": new_metrics}

    if force and not eval_promote:
        logger.warning(
            "⚠️  force=True — bypassing evaluate.py promotion gate. "
            "This model did NOT pass quality thresholds."
        )

    # -------------------------------------------------------------------------
    # STEP 3 — Compare against current production
    # -------------------------------------------------------------------------
    logger.info("Step 3/5 — Comparing against current production model …")

    registry    = _load_registry()
    prod_info   = _load_production_model_info()
    prod_metrics = prod_info.get("metrics") if prod_info else None

    if force:
        should_promote = True
        reason = "Force promotion requested — skipping comparison (force=True)"
        logger.warning("⚠️  %s", reason)
    else:
        should_promote, reason = _is_better_than_production(new_metrics, prod_metrics)

    # -------------------------------------------------------------------------
    # STEP 4 — Promote or reject
    # -------------------------------------------------------------------------
    logger.info("Step 4/5 — Making promotion decision …")

    if not should_promote:
        logger.warning("❌  Registration REJECTED: %s", reason)
        _write_audit_entry("NOT_PROMOTED", version_tag, reason, new_metrics)
        _record_in_registry(version_tag, "not_promoted", eval_report, reason, registry)
        _save_registry(registry)
        return {"promoted": False, "version_tag": version_tag,
                "reason": reason, "metrics": new_metrics}

    # --- Proceed with promotion ---
    logger.info("✅  Promoting model v%s to production …", version_tag)

    # Archive the current production model in the registry
    _archive_current_production(registry)

    # Update registry entry for the new model
    _record_in_registry(version_tag, "production", eval_report, reason, registry)

    # Update the pointer to which version is production
    registry["production_version"] = version_tag
    registry["last_promoted_at"]   = datetime.utcnow().isoformat()

    # -------------------------------------------------------------------------
    # STEP 5 — Persist everything
    # -------------------------------------------------------------------------
    logger.info("Step 5/5 — Persisting registry + symlinks …")

    # Write production_model.json — predict.py reads this on startup
    production_info = {
        "version_tag":   version_tag,
        "promoted_at":   datetime.utcnow().isoformat(),
        "model_path":    str(MODEL_DIR / f"model_v{version_tag}.pkl"),
        "pipeline_path": str(MODEL_DIR / f"pipeline_v{version_tag}.pkl"),
        "metrics":       new_metrics,
        "reason":        reason,
        "forced":        force,
    }
    with open(PRODUCTION_FILE, "w") as f:
        json.dump(production_info, f, indent=2, default=str)
    logger.info("Production model info saved → %s", PRODUCTION_FILE)

    # Update the registry JSON
    _save_registry(registry)

    # Update symlinks (latest_model.pkl / latest_pipeline.pkl)
    _update_symlinks(version_tag)

    # Write to the permanent audit trail
    _write_audit_entry(
        "PROMOTED", version_tag, reason, new_metrics
    )

    logger.info("=" * 60)
    logger.info(
        "✅  Model v%s is now PRODUCTION  |  roc_auc=%.4f  f1=%.4f",
        version_tag, new_metrics["roc_auc"], new_metrics["f1"],
    )
    logger.info("=" * 60)

    return {
        "promoted":    True,
        "version_tag": version_tag,
        "reason":      reason,
        "metrics":     new_metrics,
    }


# =============================================================================
# ROLLBACK FUNCTION
# =============================================================================

def rollback(target_version_tag: str) -> dict:
    """
    Rolls back production to a previously archived model version.

    When to use:
      - New model has a silent bug caught in production monitoring
      - Drift detection fires unexpectedly after a recent promotion
      - Business decision to revert while investigating an issue

    This is a MANUAL operation — call from CLI or Airflow on-demand DAG.
    It does NOT re-run evaluate.py; the target version was already evaluated.

    Args:
        target_version_tag: The version to restore as production

    Returns:
        Dict with promoted=True and rollback context
    """
    _ensure_dirs()

    logger.info("=" * 60)
    logger.info("ROLLBACK requested → version: %s", target_version_tag)
    logger.info("=" * 60)

    registry = _load_registry()

    # Check the target version exists in the registry
    if target_version_tag not in registry.get("models", {}):
        raise ValueError(
            f"Version {target_version_tag} not found in registry. "
            "Cannot roll back to an unknown version."
        )

    target_entry = registry["models"][target_version_tag]

    # Check the model artifacts still exist on disk
    model_path    = MODEL_DIR / f"model_v{target_version_tag}.pkl"
    pipeline_path = MODEL_DIR / f"pipeline_v{target_version_tag}.pkl"

    if not model_path.exists() or not pipeline_path.exists():
        raise FileNotFoundError(
            f"Model or pipeline artifacts for version {target_version_tag} "
            "not found on disk. Cannot roll back."
        )

    # Archive the current production model first
    _archive_current_production(registry)

    # Restore target version as production
    registry["models"][target_version_tag]["status"]      = "production"
    registry["models"][target_version_tag]["restored_at"] = datetime.utcnow().isoformat()
    registry["production_version"] = target_version_tag
    registry["last_promoted_at"]   = datetime.utcnow().isoformat()

    # Update production_model.json
    target_metrics = target_entry.get("metrics", {})
    production_info = {
        "version_tag":   target_version_tag,
        "promoted_at":   datetime.utcnow().isoformat(),
        "model_path":    str(model_path),
        "pipeline_path": str(pipeline_path),
        "metrics":       target_metrics,
        "reason":        "Rollback from CLI/Airflow",
        "forced":        True,
        "is_rollback":   True,
    }
    with open(PRODUCTION_FILE, "w") as f:
        json.dump(production_info, f, indent=2, default=str)

    _save_registry(registry)
    _update_symlinks(target_version_tag)

    reason = f"Manual rollback to v{target_version_tag}"
    _write_audit_entry("ROLLBACK", target_version_tag, reason, target_metrics)

    logger.info("=" * 60)
    logger.info("✅  Rollback complete — production is now v%s", target_version_tag)
    logger.info("=" * 60)

    return {
        "promoted":    True,
        "version_tag": target_version_tag,
        "reason":      reason,
        "metrics":     target_metrics,
        "is_rollback": True,
    }


# =============================================================================
# REGISTRY RECORD HELPER
# =============================================================================

def _record_in_registry(version_tag: str, status: str,
                         eval_report: dict, reason: str,
                         registry: Optional[dict] = None) -> None:
    """
    Adds or updates a model's entry in the registry dict.

    Called for all outcomes — promoted, rejected, failed — so the registry
    is a complete history of every model that was ever trained and evaluated.

    Args:
        version_tag:  The model version
        status:       'production' | 'archived' | 'failed' | 'not_promoted'
        eval_report:  Full eval report dict from evaluate.py
        reason:       Human-readable explanation of the decision
        registry:     Registry dict to update in-place (loads fresh if None)
    """
    if registry is None:
        registry = _load_registry()

    registry["models"][version_tag] = {
        "version_tag":    version_tag,
        "status":         status,
        "recorded_at":    datetime.utcnow().isoformat(),
        "reason":         reason,
        "metrics":        eval_report.get("metrics", {}),
        "promote_flag":   eval_report.get("promote"),
        "promotion_gate": eval_report.get("promotion_gate", {}),
        "training_context": eval_report.get("training_context", {}),
        "model_path":     str(MODEL_DIR / f"model_v{version_tag}.pkl"),
        "pipeline_path":  str(MODEL_DIR / f"pipeline_v{version_tag}.pkl"),
        "eval_report_path": str(MODEL_DIR / f"eval_report_v{version_tag}.json"),
    }


# =============================================================================
# LIST MODELS HELPER
# =============================================================================

def list_models(status_filter: Optional[str] = None) -> list:
    """
    Returns all registered model versions, optionally filtered by status.

    Useful for:
      - Airflow UI / monitoring dashboards
      - Picking a rollback target
      - Auditing how many models were trained vs promoted

    Args:
        status_filter: 'production' | 'archived' | 'failed' | None (all)

    Returns:
        List of model entry dicts, sorted newest first
    """
    registry = _load_registry()
    models   = list(registry.get("models", {}).values())

    if status_filter:
        models = [m for m in models if m.get("status") == status_filter]

    # Sort by recorded_at descending (newest first)
    models.sort(key=lambda m: m.get("recorded_at", ""), reverse=True)
    return models


# =============================================================================
# CLI ENTRY POINT
# =============================================================================
# Usage examples:
#
#   Promote:
#     python -m training.register_model --version_tag 20240415_143022
#
#   Force promote (skip comparison):
#     python -m training.register_model --version_tag 20240415_143022 --force
#
#   Rollback:
#     python -m training.register_model --action rollback --version_tag 20240101_090000
#
#   List all models:
#     python -m training.register_model --action list
#
#   List only production model:
#     python -m training.register_model --action list --status production
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Register or rollback ML model versions.")
    parser.add_argument(
        "--action", type=str, default="promote",
        choices=["promote", "rollback", "list"],
        help="Action to perform (default: promote)"
    )
    parser.add_argument(
        "--version_tag", type=str, default=None,
        help="Model version tag, e.g. 20240415_143022"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force promote even if comparison check fails"
    )
    parser.add_argument(
        "--status", type=str, default=None,
        choices=["production", "archived", "failed", "not_promoted"],
        help="Filter for --action list"
    )
    args = parser.parse_args()

    if args.action == "promote":
        if not args.version_tag:
            parser.error("--version_tag is required for promote")
        result = register_model(args.version_tag, force=args.force)
        print(f"\n  Promoted : {result['promoted']}")
        print(f"  Version  : {result['version_tag']}")
        print(f"  Reason   : {result['reason']}")
        if result.get("metrics"):
            m = result["metrics"]
            print(f"  Metrics  : acc={m.get('accuracy','n/a'):.4f}  "
                  f"f1={m.get('f1','n/a'):.4f}  roc_auc={m.get('roc_auc','n/a'):.4f}")

    elif args.action == "rollback":
        if not args.version_tag:
            parser.error("--version_tag is required for rollback")
        result = rollback(args.version_tag)
        print(f"\n  Rollback complete → production is now v{result['version_tag']}")

    elif args.action == "list":
        models = list_models(status_filter=args.status)
        print(f"\n{'VERSION TAG':<22} {'STATUS':<15} {'ROC-AUC':<10} {'F1':<10} RECORDED AT")
        print("-" * 80)
        for m in models:
            metrics = m.get("metrics", {})
            print(
                f"{m['version_tag']:<22} "
                f"{m.get('status','unknown'):<15} "
                f"{metrics.get('roc_auc', 'n/a'):<10} "
                f"{metrics.get('f1', 'n/a'):<10} "
                f"{m.get('recorded_at','')}"
            )
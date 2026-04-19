"""
airflow/dags/drift_check_dag.py
Periodic drift + quality check on live prediction window:
  load_reference → fetch_live_window → compute_drift → compute_quality → push_metrics → gate

Schedule  : DRIFT_CHECK_SCHEDULE env (default: every 6 hours)
XCom      : drift_detected (bool), quality_passed (bool), report paths
TriggerRule: gate task sets XCom read by retrain_trigger_dag via Airflow Dataset
             or external trigger check (see retrain_trigger_dag.py)
"""

from __future__ import annotations

import logging
import os
from datetime import timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.dates import days_ago

log = logging.getLogger(__name__)

SCHEDULE         = os.getenv("DRIFT_CHECK_SCHEDULE", "0 */6 * * *")
WINDOW_HOURS     = int(os.getenv("DRIFT_WINDOW_HOURS", "6"))
MIN_WINDOW_ROWS  = int(os.getenv("DRIFT_MIN_SAMPLES", "30"))
ARTIFACTS_DIR    = os.getenv("ARTIFACTS_DIR", "artifacts")
RETRAIN_VAR_KEY  = "drift_retrain_triggered"   # Airflow Variable set by gate task

DEFAULT_ARGS = {
    "owner": "ml-platform",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=3),
    "execution_timeout": timedelta(minutes=30),
}


# ── Task callables ────────────────────────────────────────────────────────────

def task_load_reference(**ctx):
    """Load training reference dataset from latest registered model artifacts."""
    import json
    import pandas as pd
    from pathlib import Path

    reg_path = Path(ARTIFACTS_DIR) / "model_registry.json"
    if not reg_path.exists():
        raise FileNotFoundError(f"Model registry not found: {reg_path}")

    with open(reg_path) as f:
        registry = json.load(f)

    # Find production model entry
    prod = next((m for m in registry.get("models", []) if m.get("status") == "production"), None)
    if prod is None:
        raise RuntimeError("No production model in registry.")

    tag = prod["version_tag"]
    ref_path = Path(ARTIFACTS_DIR) / f"features_{tag}.parquet"

    # Fallback: any features parquet for that version
    if not ref_path.exists():
        candidates = sorted(Path(ARTIFACTS_DIR).glob("features_*.parquet"), reverse=True)
        if not candidates:
            raise FileNotFoundError("No reference features parquet found.")
        ref_path = candidates[0]
        log.warning("Exact reference not found, using fallback: %s", ref_path)

    ref_df = pd.read_parquet(ref_path)
    ti = ctx["ti"]
    ti.xcom_push(key="ref_path",      value=str(ref_path))
    ti.xcom_push(key="model_version", value=tag)
    ti.xcom_push(key="ref_size",      value=len(ref_df))
    log.info("Reference loaded: %d rows  model_version=%s", len(ref_df), tag)


def task_fetch_live_window(**ctx):
    """Pull live prediction logs from the last WINDOW_HOURS via logger.query_logs."""
    import pandas as pd
    from datetime import datetime, timezone

    ti = ctx["ti"]
    now = datetime.now(timezone.utc)
    window_start = (now - timedelta(hours=WINDOW_HOURS)).isoformat()
    window_end   = now.isoformat()

    try:
        from api.logger import get_logger
        records = get_logger().query_logs(start=window_start, end=window_end, limit=50_000)
        df = pd.DataFrame(records)
    except Exception as e:
        log.warning("Logger unavailable (%s). Attempting JSONL fallback.", e)
        import glob, json
        from pathlib import Path
        rows = []
        for f in sorted(glob.glob(f"{ARTIFACTS_DIR}/logs/*.jsonl"), reverse=True)[:7]:
            with open(f) as fh:
                for line in fh:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        df = pd.DataFrame(rows)
        if "timestamp" in df.columns:
            df = df[df["timestamp"] >= window_start]

    if len(df) < MIN_WINDOW_ROWS:
        raise ValueError(
            f"Live window has only {len(df)} rows (need ≥ {MIN_WINDOW_ROWS}). "
            "Skipping drift check."
        )

    live_path = f"{ARTIFACTS_DIR}/live_window_{ctx['ds']}.parquet"
    df.to_parquet(live_path, index=False)

    ti.xcom_push(key="live_path",    value=live_path)
    ti.xcom_push(key="window_start", value=window_start)
    ti.xcom_push(key="window_end",   value=window_end)
    ti.xcom_push(key="live_size",    value=len(df))
    log.info("Live window: %d rows [%s → %s]", len(df), window_start, window_end)


def task_compute_drift(**ctx):
    import pandas as pd
    from monitoring.drift_report import compute_drift_report

    ti = ctx["ti"]
    ref_df   = pd.read_parquet(ti.xcom_pull(task_ids="load_reference",    key="ref_path"))
    live_df  = pd.read_parquet(ti.xcom_pull(task_ids="fetch_live_window", key="live_path"))
    version  = ti.xcom_pull(task_ids="load_reference",    key="model_version")
    w_start  = ti.xcom_pull(task_ids="fetch_live_window", key="window_start")
    w_end    = ti.xcom_pull(task_ids="fetch_live_window", key="window_end")

    # Separate feature cols from log metadata
    meta_cols = {"request_id", "predicted_class", "confidence", "model_version", "timestamp"}
    feature_cols = [c for c in live_df.columns if c not in meta_cols and c in ref_df.columns]

    report = compute_drift_report(
        reference_df=ref_df[feature_cols],
        current_df=live_df[feature_cols],
        model_version=version,
        window_start=w_start,
        window_end=w_end,
        ref_logs=ref_df,
        cur_logs=live_df,
        save=True,
    )

    ti.xcom_push(key="drift_detected",    value=report.overall_drifted)
    ti.xcom_push(key="drifted_features",  value=report.drifted_features)
    ti.xcom_push(key="drift_report_id",   value=report.report_id)
    ti.xcom_push(key="critical_count",    value=report.summary.get("critical_count", 0))
    log.info("Drift: drifted=%s features=%s critical=%d",
             report.overall_drifted, report.drifted_features, report.summary.get("critical_count", 0))


def task_compute_quality(**ctx):
    import pandas as pd
    from monitoring.quality_report import compute_quality_report

    ti = ctx["ti"]
    live_df = pd.read_parquet(ti.xcom_pull(task_ids="fetch_live_window", key="live_path"))
    version = ti.xcom_pull(task_ids="load_reference", key="model_version")
    w_start = ti.xcom_pull(task_ids="fetch_live_window", key="window_start")
    w_end   = ti.xcom_pull(task_ids="fetch_live_window", key="window_end")

    report = compute_quality_report(
        log_df=live_df,
        model_version=version,
        window_start=w_start,
        window_end=w_end,
        save=True,
    )

    ti.xcom_push(key="quality_passed",    value=report.overall_passed)
    ti.xcom_push(key="hard_failures",     value=report.hard_failures)
    ti.xcom_push(key="quality_report_id", value=report.report_id)
    log.info("Quality: passed=%s hard_failures=%s", report.overall_passed, report.hard_failures)


def task_push_metrics(**ctx):
    """Refresh Prometheus metrics from latest reports."""
    try:
        from monitoring.prometheus_exporter import collect_all_metrics
        text = collect_all_metrics()
        log.info("Prometheus metrics refreshed (%d bytes)", len(text))
    except Exception as e:
        # Non-fatal — exporter may be running in separate process
        log.warning("Metrics push skipped: %s", e)


def task_gate(**ctx):
    """
    Write drift/quality outcomes to Airflow Variable.
    retrain_trigger_dag polls this variable to decide whether to fire.
    """
    from airflow.models import Variable

    ti = ctx["ti"]
    drift_detected = ti.xcom_pull(task_ids="compute_drift",   key="drift_detected") or False
    quality_passed = ti.xcom_pull(task_ids="compute_quality", key="quality_passed")
    critical_count = ti.xcom_pull(task_ids="compute_drift",   key="critical_count") or 0

    should_retrain = drift_detected or (quality_passed is False)

    Variable.set(RETRAIN_VAR_KEY, str(should_retrain).lower())
    Variable.set("drift_critical_count", str(critical_count))
    Variable.set("last_drift_check_ts", ctx["ts"])

    log.info(
        "Gate: drift=%s quality_passed=%s critical=%d → retrain=%s",
        drift_detected, quality_passed, critical_count, should_retrain,
    )
    ti.xcom_push(key="should_retrain", value=should_retrain)

    if should_retrain:
        _on_drift_alert(ctx, drift_detected, quality_passed, critical_count)


def _on_drift_alert(ctx, drift_detected, quality_passed, critical_count):
    try:
        from alerting.notify import send_alert
        lines = []
        if drift_detected:
            features = ctx["ti"].xcom_pull(task_ids="compute_drift", key="drifted_features") or []
            lines.append(f"Drift detected on: {features}  (critical={critical_count})")
        if not quality_passed:
            failures = ctx["ti"].xcom_pull(task_ids="compute_quality", key="hard_failures") or []
            lines.append(f"Quality failures: {failures}")
        send_alert(
            channel="slack",
            title="[DRIFT CHECK] Retraining triggered",
            message="\n".join(lines),
            severity="warning" if critical_count == 0 else "critical",
            labels={"dag": "ml_drift_check", "ds": ctx["ds"]},
        )
    except Exception as e:
        log.error("Drift alert send failed: %s", e)


def _on_failure(context):
    try:
        from alerting.notify import send_alert
        ti = context["task_instance"]
        send_alert(
            channel="slack",
            title=f"[DRIFT CHECK FAIL] {ti.task_id}",
            message=str(context.get("exception", "unknown")),
            severity="critical",
            labels={"dag": "ml_drift_check", "task": ti.task_id},
        )
    except Exception as e:
        log.error("Alert send failed: %s", e)


# ── DAG ───────────────────────────────────────────────────────────────────────

with DAG(
    dag_id="ml_drift_check",
    description="Periodic drift + quality check on live prediction window",
    schedule_interval=SCHEDULE,
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
    tags=["ml", "monitoring", "drift"],
    default_args={**DEFAULT_ARGS, "on_failure_callback": _on_failure},
    doc_md=__doc__,
) as dag:

    t_ref     = PythonOperator(task_id="load_reference",    python_callable=task_load_reference)
    t_live    = PythonOperator(task_id="fetch_live_window", python_callable=task_fetch_live_window)
    t_drift   = PythonOperator(task_id="compute_drift",     python_callable=task_compute_drift)
    t_quality = PythonOperator(task_id="compute_quality",   python_callable=task_compute_quality)
    t_metrics = PythonOperator(task_id="push_metrics",      python_callable=task_push_metrics)
    t_gate    = PythonOperator(task_id="gate",              python_callable=task_gate)

    t_ref >> t_live >> [t_drift, t_quality] >> t_metrics >> t_gate
"""
airflow/dags/training_dag.py
Full training pipeline:
  ingest → validate → build_features → train → evaluate → register_model

Schedule : TRAINING_SCHEDULE env (default: weekly Sunday 02:00)
XCom     : artifact paths + metrics passed between tasks
Alerts   : on_failure_callback → alerting/notify.py
"""

from __future__ import annotations

import json
import logging
import os
from datetime import timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

log = logging.getLogger(__name__)

SCHEDULE        = os.getenv("TRAINING_SCHEDULE", "0 2 * * 0")
DATA_SOURCE     = os.getenv("TRAINING_DATA_SOURCE", "db")
MODEL_VERSION   = os.getenv("MODEL_VERSION_TAG", "")
PROMOTE_TO_PROD = os.getenv("AUTO_PROMOTE", "false").lower() == "true"
ARTIFACTS_DIR   = os.getenv("ARTIFACTS_DIR", "artifacts")

DEFAULT_ARGS = {
    "owner": "ml-platform",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=2),
}


# ── Task callables ────────────────────────────────────────────────────────────

def task_ingest(**ctx):
    from data.ingest import ingest_data
    ds = ctx["ds"]
    df = ingest_data(source=DATA_SOURCE, date_partition=ds)
    path = f"{ARTIFACTS_DIR}/ingested_{ds}.parquet"
    df.to_parquet(path, index=False)
    ctx["ti"].xcom_push(key="ingested_path", value=path)
    ctx["ti"].xcom_push(key="row_count", value=len(df))
    log.info("Ingested %d rows → %s", len(df), path)


def task_validate(**ctx):
    import pandas as pd
    from data.validate import validate_dataframe

    ti = ctx["ti"]
    df = pd.read_parquet(ti.xcom_pull(task_ids="ingest", key="ingested_path"))
    report = validate_dataframe(df)

    hard_fails = [c for c in report.checks if not c.passed and c.severity == "hard"]
    if hard_fails:
        raise ValueError(
            f"Validation: {len(hard_fails)} hard failure(s): "
            + ", ".join(c.check_name for c in hard_fails)
        )
    soft = len([c for c in report.checks if not c.passed and c.severity == "soft"])
    ti.xcom_push(key="soft_warnings", value=soft)
    log.info("Validation passed (soft warnings: %d)", soft)


def task_build_features(**ctx):
    import pandas as pd
    from data.features import FeaturePipeline

    ti, ds = ctx["ti"], ctx["ds"]
    df = pd.read_parquet(ti.xcom_pull(task_ids="ingest", key="ingested_path"))

    pipeline = FeaturePipeline()
    X = pipeline.fit_transform(df)

    feat_path = f"{ARTIFACTS_DIR}/features_{ds}.parquet"
    pipe_path = f"{ARTIFACTS_DIR}/pipeline_{ds}.pkl"
    X.to_parquet(feat_path, index=False)
    pipeline.save(pipe_path)

    ti.xcom_push(key="features_path", value=feat_path)
    ti.xcom_push(key="pipeline_path", value=pipe_path)
    log.info("Features: %d × %d → %s", len(X), X.shape[1], feat_path)


def task_train(**ctx):
    from training.train import run_training

    ti, ds = ctx["ti"], ctx["ds"]
    tag = MODEL_VERSION or ds.replace("-", "")
    result = run_training(
        features_path=ti.xcom_pull(task_ids="build_features", key="features_path"),
        pipeline_path=ti.xcom_pull(task_ids="build_features", key="pipeline_path"),
        version_tag=tag,
        artifacts_dir=ARTIFACTS_DIR,
    )
    ti.xcom_push(key="model_path",   value=result["model_path"])
    ti.xcom_push(key="version_tag",  value=result["version_tag"])
    ti.xcom_push(key="val_accuracy", value=result["val_accuracy"])
    log.info("Trained v%s  val_acc=%.4f", result["version_tag"], result["val_accuracy"])


def task_evaluate(**ctx):
    from training.evaluate import run_evaluation

    ti = ctx["ti"]
    report = run_evaluation(
        model_path=ti.xcom_pull(task_ids="train",         key="model_path"),
        features_path=ti.xcom_pull(task_ids="build_features", key="features_path"),
        version_tag=ti.xcom_pull(task_ids="train",        key="version_tag"),
        artifacts_dir=ARTIFACTS_DIR,
    )
    if not report.get("promotion_gate_passed", False):
        metrics_str = json.dumps({k: round(v, 4) for k, v in report.get("metrics", {}).items()})
        raise ValueError(f"Promotion gate failed. metrics={metrics_str}")

    ti.xcom_push(key="eval_report_path", value=report["eval_report_path"])
    ti.xcom_push(key="metrics",          value=report.get("metrics", {}))
    m = report.get("metrics", {})
    log.info("Eval passed. F1=%.4f AUC=%.4f", m.get("f1", 0), m.get("roc_auc", 0))


def task_register(**ctx):
    from training.register_model import register_model

    ti = ctx["ti"]
    result = register_model(
        version_tag=ti.xcom_pull(task_ids="train",    key="version_tag"),
        model_path=ti.xcom_pull(task_ids="train",     key="model_path"),
        eval_report_path=ti.xcom_pull(task_ids="evaluate", key="eval_report_path"),
        promote_to_production=PROMOTE_TO_PROD,
        artifacts_dir=ARTIFACTS_DIR,
    )
    ti.xcom_push(key="registered", value=result["registered"])
    ti.xcom_push(key="promoted",   value=result.get("promoted", False))
    log.info("Registered. promoted=%s", result.get("promoted"))


def _on_failure(context):
    try:
        from alerting.notify import send_alert
        ti = context["task_instance"]
        send_alert(
            channel="slack",
            title=f"[TRAINING FAIL] {context['dag'].dag_id} / {ti.task_id}",
            message=str(context.get("exception", "unknown error")),
            severity="critical",
            labels={"dag": context["dag"].dag_id, "task": ti.task_id},
        )
    except Exception as e:
        log.error("Alert send failed: %s", e)


# ── DAG ───────────────────────────────────────────────────────────────────────

with DAG(
    dag_id="ml_training_pipeline",
    description="ingest → validate → features → train → evaluate → register",
    schedule_interval=SCHEDULE,
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
    tags=["ml", "training"],
    default_args={**DEFAULT_ARGS, "on_failure_callback": _on_failure},
    doc_md=__doc__,
) as dag:

    t_ingest   = PythonOperator(task_id="ingest",          python_callable=task_ingest)
    t_validate = PythonOperator(task_id="validate",        python_callable=task_validate)
    t_features = PythonOperator(task_id="build_features",  python_callable=task_build_features)
    t_train    = PythonOperator(task_id="train",           python_callable=task_train)
    t_evaluate = PythonOperator(task_id="evaluate",        python_callable=task_evaluate)
    t_register = PythonOperator(task_id="register",        python_callable=task_register)

    t_ingest >> t_validate >> t_features >> t_train >> t_evaluate >> t_register
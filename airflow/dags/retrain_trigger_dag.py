"""
airflow/dags/retrain_trigger_dag.py
Polls drift_check_dag XCom for the `should_retrain` flag and conditionally
triggers the training pipeline via Airflow's TriggerDagRunOperator.

Schedule  : Every 6 hours, offset +30 min from drift_check (runs after it)
            Configurable via RETRAIN_TRIGGER_SCHEDULE env-var.

Logic
─────
  1. poll_retrain_signal  — Read `severity_result` XCom from latest drift_check run
  2. check_cooldown       — Skip if a retrain ran within RETRAIN_COOLDOWN_HOURS
  3. branch               — route to trigger_training | skip_retrain
  4. trigger_training     — Fire ml_training_pipeline DAG run
  5. notify_retrain       — Alert team that a retrain was kicked off
  6. update_cooldown      — Write cooldown timestamp to shared state file

Guard rails
───────────
  - Cooldown window prevents retrain storms (default 12 h)
  - Max retrain runs per day cap (RETRAIN_MAX_PER_DAY, default 2)
  - Manual override: set FORCE_RETRAIN=true in Airflow Variable
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from airflow import DAG
from airflow.models import DagRun, Variable
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.utils.dates import days_ago
from airflow.utils.state import DagRunState
from airflow.utils.trigger_rule import TriggerRule

log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

SCHEDULE = os.getenv("RETRAIN_TRIGGER_SCHEDULE", "30 */6 * * *")
ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "artifacts")
DRIFT_DAG_ID = "ml_drift_check"
TRAINING_DAG_ID = "ml_training_pipeline"
COOLDOWN_HOURS = int(os.getenv("RETRAIN_COOLDOWN_HOURS", "12"))
MAX_PER_DAY = int(os.getenv("RETRAIN_MAX_PER_DAY", "2"))
COOLDOWN_FILE = Path(ARTIFACTS_DIR) / "retrain_cooldown.json"
FORCE_RETRAIN_VAR = "FORCE_RETRAIN"  # Airflow Variable name

DEFAULT_ARGS = {
    "owner": "ml-team",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(minutes=15),
    "on_failure_callback": None,
}


# ── Cooldown state helpers ────────────────────────────────────────────────────


def _read_cooldown() -> dict:
    if COOLDOWN_FILE.exists():
        try:
            with open(COOLDOWN_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {"last_retrain_utc": None, "runs_today": 0, "today_date": None}


def _write_cooldown(state: dict) -> None:
    COOLDOWN_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(COOLDOWN_FILE, "w") as f:
        json.dump(state, f, indent=2)


def _count_today_retrains() -> int:
    """Count successful training DAG runs today via Airflow metadata DB."""
    today = datetime.now(timezone.utc).date()
    try:
        runs = DagRun.find(
            dag_id=TRAINING_DAG_ID,
            state=DagRunState.SUCCESS,
            execution_start_date=datetime.combine(today, datetime.min.time()).replace(
                tzinfo=timezone.utc
            ),
        )
        return len(runs) if runs else 0
    except Exception as e:
        log.warning("Could not query DagRun history: %s. Using cooldown file.", e)
        state = _read_cooldown()
        today_str = str(today)
        if state.get("today_date") == today_str:
            return state.get("runs_today", 0)
        return 0


# ── Task callables ────────────────────────────────────────────────────────────


def _poll_retrain_signal(**ctx) -> dict:
    """
    Read severity_result XCom from the most recent drift_check DAG run.
    Falls back to FORCE_RETRAIN Airflow Variable for manual override.
    """
    # Check manual override first
    force = False
    try:
        force_val = Variable.get(FORCE_RETRAIN_VAR, default_var="false")
        force = str(force_val).lower() == "true"
        if force:
            log.warning("FORCE_RETRAIN Variable is set — overriding drift signal.")
    except Exception:
        pass

    signal = {
        "should_retrain": force,
        "severity": "forced" if force else "ok",
        "source": "manual_override" if force else "drift_check",
    }

    if not force:
        # Fetch latest drift_check DAG run XCom
        try:
            latest_runs = DagRun.find(dag_id=DRIFT_DAG_ID, state=DagRunState.SUCCESS)
            if latest_runs:
                latest_run = max(latest_runs, key=lambda r: r.execution_date)
                from airflow.models import TaskInstance

                tis = TaskInstance.find(
                    dag_id=DRIFT_DAG_ID,
                    run_id=latest_run.run_id,
                    task_id="evaluate_severity",
                )
                if tis:
                    severity_result = (
                        tis[0].xcom_pull(
                            task_ids="evaluate_severity",
                            key="severity_result",
                            dag_id=DRIFT_DAG_ID,
                            include_prior_dates=False,
                        )
                        or {}
                    )
                    signal = {
                        "should_retrain": severity_result.get("should_retrain", False),
                        "severity": severity_result.get("severity", "ok"),
                        "drift_report_id": severity_result.get("drift_report_id"),
                        "quality_report_id": severity_result.get("quality_report_id"),
                        "source": "drift_check_xcom",
                    }
        except Exception as e:
            log.error("XCom poll failed: %s — defaulting to no retrain.", e)

    ctx["ti"].xcom_push(key="retrain_signal", value=signal)
    log.info(
        "Retrain signal: should_retrain=%s severity=%s source=%s",
        signal["should_retrain"],
        signal["severity"],
        signal["source"],
    )
    return signal


def _check_cooldown(**ctx) -> dict:
    """
    Apply guard rails:
      - Cooldown window (COOLDOWN_HOURS since last retrain)
      - Max runs per day (MAX_PER_DAY)
    Pushes `approved` bool.
    """
    signal = ctx["ti"].xcom_pull(key="retrain_signal") or {}

    if not signal.get("should_retrain"):
        ctx["ti"].xcom_push(
            key="cooldown_result", value={"approved": False, "reason": "no_signal"}
        )
        return {"approved": False, "reason": "no_signal"}

    state = _read_cooldown()
    now = datetime.now(timezone.utc)

    # Cooldown check
    if state.get("last_retrain_utc"):
        last = datetime.fromisoformat(state["last_retrain_utc"])
        elapsed_h = (now - last).total_seconds() / 3600
        if elapsed_h < COOLDOWN_HOURS:
            reason = f"cooldown ({elapsed_h:.1f}h < {COOLDOWN_HOURS}h required)"
            log.info("Retrain suppressed: %s", reason)
            ctx["ti"].xcom_push(
                key="cooldown_result", value={"approved": False, "reason": reason}
            )
            return {"approved": False, "reason": reason}

    # Daily cap check
    today_count = _count_today_retrains()
    if today_count >= MAX_PER_DAY:
        reason = f"daily_cap ({today_count}/{MAX_PER_DAY} runs today)"
        log.info("Retrain suppressed: %s", reason)
        ctx["ti"].xcom_push(
            key="cooldown_result", value={"approved": False, "reason": reason}
        )
        return {"approved": False, "reason": reason}

    ctx["ti"].xcom_push(
        key="cooldown_result", value={"approved": True, "reason": "cleared"}
    )
    log.info("Retrain approved. Today's count: %d/%d", today_count, MAX_PER_DAY)
    return {"approved": True, "reason": "cleared"}


def _branch_retrain(**ctx) -> str:
    cooldown = ctx["ti"].xcom_pull(key="cooldown_result") or {}
    return "trigger_training" if cooldown.get("approved") else "skip_retrain"


def _notify_retrain_triggered(**ctx) -> None:
    """Alert team that an automated retrain was kicked off."""
    from alerting.notify import send_alert

    signal = ctx["ti"].xcom_pull(key="retrain_signal") or {}
    cooldown = ctx["ti"].xcom_pull(key="cooldown_result") or {}

    send_alert(
        level="warning",
        title="[Retrain Trigger] Automated retrain initiated",
        message=(
            f"Drift severity: {signal.get('severity', 'unknown')}\n"
            f"Drift report: {signal.get('drift_report_id', 'n/a')}\n"
            f"Quality report: {signal.get('quality_report_id', 'n/a')}\n"
            f"Cooldown status: {cooldown.get('reason', 'n/a')}\n"
            f"Training DAG: {TRAINING_DAG_ID} has been triggered."
        ),
        tags={
            "source": signal.get("source", "unknown"),
            "severity": signal.get("severity", "unknown"),
        },
    )
    log.info("Retrain notification sent.")


def _update_cooldown(**ctx) -> None:
    """Write updated cooldown state after successful trigger."""
    now = datetime.now(timezone.utc)
    today = str(now.date())
    state = _read_cooldown()

    # Reset daily counter if date rolled over
    if state.get("today_date") != today:
        state["runs_today"] = 0
        state["today_date"] = today

    state["last_retrain_utc"] = now.isoformat()
    state["runs_today"] = state.get("runs_today", 0) + 1
    _write_cooldown(state)
    log.info(
        "Cooldown updated: last=%s runs_today=%d", now.isoformat(), state["runs_today"]
    )


def _log_skip(**ctx) -> None:
    cooldown = ctx["ti"].xcom_pull(key="cooldown_result") or {}
    signal = ctx["ti"].xcom_pull(key="retrain_signal") or {}
    log.info(
        "Retrain skipped. should_retrain=%s reason=%s",
        signal.get("should_retrain"),
        cooldown.get("reason"),
    )


def _clear_force_retrain_var(**ctx) -> None:
    """Reset FORCE_RETRAIN variable if it was set, after retrain triggered."""
    try:
        Variable.set(FORCE_RETRAIN_VAR, "false")
    except Exception as e:
        log.warning("Could not reset %s Variable: %s", FORCE_RETRAIN_VAR, e)


def _notify_failure(context) -> None:
    try:
        from alerting.notify import send_alert

        dag_id = context["dag"].dag_id
        task_id = context["task_instance"].task_id
        send_alert(
            level="critical",
            title=f"[{dag_id}] Task '{task_id}' failed",
            message=str(context.get("exception", "unknown")),
            tags={"dag": dag_id, "task": task_id},
        )
    except Exception as e:
        log.error("Failure alert error: %s", e)


DEFAULT_ARGS["on_failure_callback"] = _notify_failure


# ── DAG ───────────────────────────────────────────────────────────────────────

with DAG(
    dag_id="ml_retrain_trigger",
    description="Polls drift_check XCom and conditionally triggers ml_training_pipeline",
    schedule_interval=SCHEDULE,
    start_date=days_ago(1),
    default_args=DEFAULT_ARGS,
    catchup=False,
    max_active_runs=1,
    tags=["ml", "retrain", "trigger"],
    doc_md=f"""
## ML Retrain Trigger DAG

Runs 30 min after drift_check. Reads its `severity_result` XCom output
and triggers `{TRAINING_DAG_ID}` if retraining is warranted.

### Guard rails
- **Cooldown**: {COOLDOWN_HOURS} hours between automated retrains
- **Daily cap**: max {MAX_PER_DAY} triggered retrains per day
- **Manual override**: Set Airflow Variable `{FORCE_RETRAIN_VAR}=true`

### Steps
1. **poll_retrain_signal** — Read XCom from latest drift_check run
2. **check_cooldown**      — Apply cooldown + daily cap guards
3. **branch_retrain**      — Route to trigger or skip
4. **trigger_training**    — Fire ml_training_pipeline DAG run
5. **notify_retrain**      — Alert team
6. **update_cooldown**     — Write new cooldown state
""",
) as dag:

    poll_signal = PythonOperator(
        task_id="poll_retrain_signal",
        python_callable=_poll_retrain_signal,
    )

    check_cooldown = PythonOperator(
        task_id="check_cooldown",
        python_callable=_check_cooldown,
    )

    branch = BranchPythonOperator(
        task_id="branch_retrain",
        python_callable=_branch_retrain,
    )

    trigger_training = TriggerDagRunOperator(
        task_id="trigger_training",
        trigger_dag_id=TRAINING_DAG_ID,
        wait_for_completion=False,  # fire-and-forget; training DAG is long-running
        reset_dag_run=False,
        conf={
            "triggered_by": "ml_retrain_trigger",
            "reason": "automated_drift_retrain",
        },
    )

    notify_triggered = PythonOperator(
        task_id="notify_retrain_triggered",
        python_callable=_notify_retrain_triggered,
    )

    update_cooldown = PythonOperator(
        task_id="update_cooldown",
        python_callable=_update_cooldown,
    )

    clear_force_var = PythonOperator(
        task_id="clear_force_retrain_var",
        python_callable=_clear_force_retrain_var,
    )

    skip_retrain = PythonOperator(
        task_id="skip_retrain",
        python_callable=_log_skip,
    )

    done = EmptyOperator(
        task_id="done",
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )

    # Pipeline
    poll_signal >> check_cooldown >> branch
    branch >> [trigger_training, skip_retrain]
    trigger_training >> notify_triggered >> update_cooldown >> clear_force_var >> done
    skip_retrain >> done

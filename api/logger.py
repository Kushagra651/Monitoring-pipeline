"""
api/logger.py
=============
Prediction request / response logger.

Responsibilities
----------------
- Log every inference request (features + result) to two sinks:
    1. Postgres table  `prediction_logs`  (primary store, queried by monitoring)
    2. Local JSONL file                   (fallback / offline replay)
- Buffer writes and flush in a background thread so the API path is never
  blocked by a slow DB write.
- Provide a `PredictionLog` dataclass that is the canonical record shape
  consumed by monitoring/quality_report.py and monitoring/drift_report.py.
- Expose `query_logs()` for time-windowed retrieval used by monitoring jobs.

Public API
----------
    log_prediction(request_id, features, result, ground_truth=None) -> None
    flush()                      -> int          # force-flush buffer, return rows written
    query_logs(start, end, ...)  -> pd.DataFrame
    PredictionLog                               # dataclass
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
# import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Config from environment
# ---------------------------------------------------------------------------
DB_URL: str = os.getenv(
    "DATABASE_URL",
    "postgresql://mluser:mlpassword@localhost:5432/mlmonitoring",
)
LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
JSONL_FILE = LOG_DIR / "predictions.jsonl"
BUFFER_FLUSH_INTERVAL: float = float(os.getenv("LOG_FLUSH_INTERVAL", "5"))   # seconds
BUFFER_MAX_SIZE: int = int(os.getenv("LOG_BUFFER_SIZE", "200"))               # rows

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FeatureSchema — graceful fallback
# ---------------------------------------------------------------------------
try:
    from api.schemas import FeatureSchema  # type: ignore
    _SCHEMA: FeatureSchema | None = FeatureSchema()
except ImportError:
    _SCHEMA = None
    logger.warning("api.schemas not importable — schema metadata will be omitted from logs")

# ---------------------------------------------------------------------------
# Optional Postgres driver
# ---------------------------------------------------------------------------
try:
    import psycopg2
    import psycopg2.extras
    _PSYCOPG2_AVAILABLE = True
except ImportError:
    _PSYCOPG2_AVAILABLE = False
    logger.warning("psycopg2 not installed — DB logging disabled, falling back to JSONL only")


# ---------------------------------------------------------------------------
# PredictionLog — canonical log record
# ---------------------------------------------------------------------------

@dataclass
class PredictionLog:
    """
    Single prediction event.  Written to Postgres + JSONL.
    Consumed by monitoring/drift_report.py and monitoring/quality_report.py.
    """
    request_id: str                        # UUID, set by caller or auto-generated
    timestamp: str                         # ISO-8601 UTC
    model_version: str
    model_alias: str
    features: dict[str, Any]               # raw input features
    prediction: int                        # 0 or 1
    probability_class_0: float
    probability_class_1: float
    confidence: float
    latency_ms: float
    ground_truth: int | None = None        # filled in later via label_outcome()
    warnings: list[str] = field(default_factory=list)
    schema_version: str = ""               # from FeatureSchema, if available

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        d = asdict(self)
        d["features"] = json.dumps(self.features)   # serialise for Postgres TEXT col
        d["warnings"] = json.dumps(self.warnings)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "PredictionLog":
        d = dict(d)
        if isinstance(d.get("features"), str):
            d["features"] = json.loads(d["features"])
        if isinstance(d.get("warnings"), str):
            d["warnings"] = json.loads(d["warnings"])
        return cls(**d)


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS prediction_logs (
    request_id          TEXT PRIMARY KEY,
    timestamp           TIMESTAMPTZ NOT NULL,
    model_version       TEXT,
    model_alias         TEXT,
    features            TEXT,           -- JSON string
    prediction          SMALLINT,
    probability_class_0 DOUBLE PRECISION,
    probability_class_1 DOUBLE PRECISION,
    confidence          DOUBLE PRECISION,
    latency_ms          DOUBLE PRECISION,
    ground_truth        SMALLINT,
    warnings            TEXT,           -- JSON array string
    schema_version      TEXT
);
CREATE INDEX IF NOT EXISTS idx_pl_timestamp ON prediction_logs (timestamp);
CREATE INDEX IF NOT EXISTS idx_pl_model_version ON prediction_logs (model_version);
"""

_INSERT_SQL = """
INSERT INTO prediction_logs (
    request_id, timestamp, model_version, model_alias,
    features, prediction, probability_class_0, probability_class_1,
    confidence, latency_ms, ground_truth, warnings, schema_version
) VALUES (
    %(request_id)s, %(timestamp)s, %(model_version)s, %(model_alias)s,
    %(features)s, %(prediction)s, %(probability_class_0)s, %(probability_class_1)s,
    %(confidence)s, %(latency_ms)s, %(ground_truth)s, %(warnings)s, %(schema_version)s
)
ON CONFLICT (request_id) DO NOTHING;
"""

_UPDATE_GROUND_TRUTH_SQL = """
UPDATE prediction_logs
SET ground_truth = %(ground_truth)s
WHERE request_id = %(request_id)s;
"""


def _get_conn():
    """Open a new psycopg2 connection.  Caller is responsible for closing."""
    if not _PSYCOPG2_AVAILABLE:
        raise RuntimeError("psycopg2 not available")
    return psycopg2.connect(DB_URL)


def _ensure_table() -> bool:
    """Create prediction_logs table if it doesn't exist. Returns True on success."""
    if not _PSYCOPG2_AVAILABLE:
        return False
    try:
        conn = _get_conn()
        with conn:
            with conn.cursor() as cur:
                cur.execute(_CREATE_TABLE_SQL)
        conn.close()
        logger.info("prediction_logs table ensured")
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to ensure DB table: %s", exc)
        return False


def _write_to_db(rows: list[PredictionLog]) -> int:
    """Bulk-insert a list of PredictionLog rows. Returns count written."""
    if not _PSYCOPG2_AVAILABLE or not rows:
        return 0
    try:
        conn = _get_conn()
        with conn:
            with conn.cursor() as cur:
                psycopg2.extras.execute_batch(
                    cur,
                    _INSERT_SQL,
                    [r.to_dict() for r in rows],
                    page_size=100,
                )
        conn.close()
        return len(rows)
    except Exception as exc:  # noqa: BLE001
        logger.error("DB write failed (%d rows): %s", len(rows), exc)
        return 0


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------

def _ensure_log_dir() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def _write_to_jsonl(rows: list[PredictionLog]) -> int:
    """Append rows to the JSONL fallback file. Returns count written."""
    if not rows:
        return 0
    _ensure_log_dir()
    written = 0
    try:
        with open(JSONL_FILE, "a") as fh:
            for row in rows:
                d = asdict(row)
                fh.write(json.dumps(d) + "\n")
                written += 1
    except Exception as exc:  # noqa: BLE001
        logger.error("JSONL write failed: %s", exc)
    return written


# ---------------------------------------------------------------------------
# Background flush worker
# ---------------------------------------------------------------------------

_buffer: queue.Queue[PredictionLog] = queue.Queue()
_flush_lock = threading.Lock()
_worker_started = False
_shutdown_event = threading.Event()


def _flush_worker() -> None:
    """Background thread: drain the buffer every BUFFER_FLUSH_INTERVAL seconds."""
    logger.info("Log flush worker started (interval=%.1fs)", BUFFER_FLUSH_INTERVAL)
    while not _shutdown_event.is_set():
        _shutdown_event.wait(timeout=BUFFER_FLUSH_INTERVAL)
        _drain_buffer()
    # Final drain on shutdown
    _drain_buffer()
    logger.info("Log flush worker stopped")


def _drain_buffer() -> int:
    """Pull everything off the queue and write to DB + JSONL. Returns rows written."""
    rows: list[PredictionLog] = []
    try:
        while True:
            rows.append(_buffer.get_nowait())
    except queue.Empty:
        pass

    if not rows:
        return 0

    with _flush_lock:
        db_written = _write_to_db(rows)
        jsonl_written = _write_to_jsonl(rows)
        if db_written < len(rows):
            # DB missed some rows — they were still captured in JSONL
            logger.warning(
                "DB wrote %d/%d rows; all %d captured in JSONL",
                db_written, len(rows), jsonl_written,
            )
        else:
            logger.debug("Flushed %d log rows to DB + JSONL", len(rows))
    return len(rows)


def _start_worker() -> None:
    global _worker_started
    if not _worker_started:
        _ensure_table()
        t = threading.Thread(target=_flush_worker, name="log-flush-worker", daemon=True)
        t.start()
        _worker_started = True


# ---------------------------------------------------------------------------
# Public: log_prediction
# ---------------------------------------------------------------------------

def log_prediction(
    features: dict[str, Any],
    prediction: int,
    probability_class_0: float,
    probability_class_1: float,
    confidence: float,
    latency_ms: float,
    model_version: str,
    model_alias: str,
    request_id: str | None = None,
    ground_truth: int | None = None,
    warnings: list[str] | None = None,
) -> str:
    """
    Enqueue a prediction event for async logging.

    Parameters
    ----------
    features            : raw input dict (pre-transform)
    prediction          : 0 or 1
    probability_class_0 : P(y=0)
    probability_class_1 : P(y=1)
    confidence          : max(probabilities)
    latency_ms          : end-to-end inference time
    model_version       : version tag from PredictionResult
    model_alias         : alias (e.g. "production")
    request_id          : caller-supplied UUID; auto-generated if None
    ground_truth        : optional label (usually None at inference time)
    warnings            : list of non-fatal warnings from predict()

    Returns
    -------
    str : the request_id used for this log entry
    """
    _start_worker()

    rid = request_id or str(uuid.uuid4())
    schema_ver = getattr(_SCHEMA, "version", "") if _SCHEMA else ""

    record = PredictionLog(
        request_id=rid,
        timestamp=datetime.now(timezone.utc).isoformat(),
        model_version=model_version,
        model_alias=model_alias,
        features=features,
        prediction=prediction,
        probability_class_0=probability_class_0,
        probability_class_1=probability_class_1,
        confidence=confidence,
        latency_ms=latency_ms,
        ground_truth=ground_truth,
        warnings=warnings or [],
        schema_version=schema_ver,
    )

    try:
        _buffer.put_nowait(record)
    except queue.Full:
        # Buffer overflow — write synchronously to JSONL as emergency fallback
        logger.warning("Log buffer full — writing request_id=%s directly to JSONL", rid)
        _write_to_jsonl([record])

    return rid


def log_prediction_from_result(
    features: dict[str, Any],
    result,                        # PredictionResult from predict.py
    request_id: str | None = None,
    ground_truth: int | None = None,
) -> str:
    """
    Convenience wrapper: accepts a PredictionResult object directly.
    Used by main.py so it doesn't have to unpack fields manually.
    """
    return log_prediction(
        features=features,
        prediction=result.prediction,
        probability_class_0=result.probability_class_0,
        probability_class_1=result.probability_class_1,
        confidence=result.confidence,
        latency_ms=result.latency_ms,
        model_version=result.model_version,
        model_alias=result.model_alias,
        request_id=request_id,
        ground_truth=ground_truth,
        warnings=result.warnings,
    )


# ---------------------------------------------------------------------------
# Public: label_outcome  (called when ground truth becomes available later)
# ---------------------------------------------------------------------------

def label_outcome(request_id: str, ground_truth: int) -> bool:
    """
    Update the ground_truth column for an already-logged prediction.
    Used by feedback / labelling pipelines.

    Returns True if the DB update succeeded, False otherwise.
    """
    if not _PSYCOPG2_AVAILABLE:
        logger.warning("label_outcome: DB unavailable — cannot update request_id=%s", request_id)
        return False
    try:
        conn = _get_conn()
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    _UPDATE_GROUND_TRUTH_SQL,
                    {"request_id": request_id, "ground_truth": ground_truth},
                )
                updated = cur.rowcount
        conn.close()
        if updated == 0:
            logger.warning("label_outcome: request_id=%s not found in DB", request_id)
        return updated > 0
    except Exception as exc:  # noqa: BLE001
        logger.error("label_outcome failed for request_id=%s: %s", request_id, exc)
        return False


# ---------------------------------------------------------------------------
# Public: flush  (called on graceful shutdown, or by tests)
# ---------------------------------------------------------------------------

def flush() -> int:
    """
    Force-flush the in-memory buffer to DB + JSONL immediately.
    Returns the number of rows written.
    """
    return _drain_buffer()


def shutdown() -> None:
    """Signal the background worker to stop and do a final flush."""
    _shutdown_event.set()


# ---------------------------------------------------------------------------
# Public: query_logs  (used by monitoring jobs)
# ---------------------------------------------------------------------------

def query_logs(
    start: datetime,
    end: datetime,
    model_version: str | None = None,
    include_ground_truth_only: bool = False,
    limit: int = 100_000,
) -> pd.DataFrame:
    """
    Retrieve prediction logs from Postgres for a given time window.

    Falls back to reading the JSONL file if DB is unavailable.

    Parameters
    ----------
    start, end                  : UTC datetimes (inclusive)
    model_version               : optional filter
    include_ground_truth_only   : if True, only rows where ground_truth IS NOT NULL
    limit                       : max rows returned

    Returns
    -------
    pd.DataFrame with all PredictionLog columns.
    features column is parsed back from JSON string → dict.
    """
    if _PSYCOPG2_AVAILABLE:
        return _query_logs_db(start, end, model_version, include_ground_truth_only, limit)
    else:
        logger.warning("query_logs: DB unavailable — reading from JSONL fallback")
        return _query_logs_jsonl(start, end, model_version, include_ground_truth_only, limit)


def _query_logs_db(
    start: datetime,
    end: datetime,
    model_version: str | None,
    include_ground_truth_only: bool,
    limit: int,
) -> pd.DataFrame:
    filters = ["timestamp BETWEEN %(start)s AND %(end)s"]
    params: dict[str, Any] = {"start": start, "end": end, "limit": limit}

    if model_version:
        filters.append("model_version = %(model_version)s")
        params["model_version"] = model_version

    if include_ground_truth_only:
        filters.append("ground_truth IS NOT NULL")

    where = " AND ".join(filters)
    sql = f"SELECT * FROM prediction_logs WHERE {where} ORDER BY timestamp LIMIT %(limit)s"

    try:
        conn = _get_conn()
        df = pd.read_sql(sql, conn, params=params)
        conn.close()
        # Parse features JSON string back to dict
        if "features" in df.columns:
            df["features"] = df["features"].apply(
                lambda x: json.loads(x) if isinstance(x, str) else x
            )
        if "warnings" in df.columns:
            df["warnings"] = df["warnings"].apply(
                lambda x: json.loads(x) if isinstance(x, str) else x
            )
        return df
    except Exception as exc:  # noqa: BLE001
        logger.error("query_logs DB query failed: %s", exc)
        return pd.DataFrame()


def _query_logs_jsonl(
    start: datetime,
    end: datetime,
    model_version: str | None,
    include_ground_truth_only: bool,
    limit: int,
) -> pd.DataFrame:
    if not JSONL_FILE.exists():
        logger.warning("JSONL log file not found: %s", JSONL_FILE)
        return pd.DataFrame()

    rows = []
    with open(JSONL_FILE, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            ts_str = rec.get("timestamp", "")
            try:
                ts = datetime.fromisoformat(ts_str)
                # Make aware if naive
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
            except ValueError:
                continue

            if not (start <= ts <= end):
                continue
            if model_version and rec.get("model_version") != model_version:
                continue
            if include_ground_truth_only and rec.get("ground_truth") is None:
                continue

            rows.append(rec)
            if len(rows) >= limit:
                break

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Module self-test  (python -m api.logger)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    # Minimal PredictionResult stub
    class _FakeResult:
        prediction = 1
        probability_class_0 = 0.3
        probability_class_1 = 0.7
        confidence = 0.7
        latency_ms = 4.2
        model_version = "v_test"
        model_alias = "production"
        warnings = []

    features = {"age": 34, "income": 55000, "score": 0.82}

    rid = log_prediction_from_result(features, _FakeResult())
    logger.info("Logged request_id: %s", rid)

    # Force flush to JSONL (DB likely unavailable in dev)
    n = flush()
    logger.info("Flushed %d rows", n)

    # Verify JSONL was written
    assert JSONL_FILE.exists(), f"JSONL file not created at {JSONL_FILE}"
    with open(JSONL_FILE) as fh:
        lines = [line for line in fh if line.strip()]
    assert any(rid in line for line in lines), "request_id not found in JSONL"
    logger.info("JSONL contains %d log entries", len(lines))

    # Query back
    from datetime import timedelta
    now = datetime.now(timezone.utc)
    df = query_logs(now - timedelta(minutes=1), now + timedelta(minutes=1))
    logger.info("query_logs() returned %d rows", len(df))

    logger.info("ALL SELF-TESTS PASSED")
    sys.exit(0)
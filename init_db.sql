-- init_db.sql
-- Postgres schema for ML Monitoring.
-- Executed once by the postgres container on first start
-- (mounted at /docker-entrypoint-initdb.d/).
--
-- Tables
-- ──────
--   prediction_logs     : Every prediction the API serves
--   drift_reports       : Summary row per drift check run
--   drift_feature_results: Per-feature drift metrics
--   quality_reports     : Summary row per quality check run
--   quality_checks      : Per-check results per quality run
--   model_registry      : Mirrors artifacts/model_registry.json in SQL
--   alert_log           : Append-only record of every alert fired

-- ── Extensions ────────────────────────────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ── Schemas ───────────────────────────────────────────────────────────────────
CREATE SCHEMA IF NOT EXISTS ml;
SET search_path TO ml, public;

-- ── prediction_logs ───────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS prediction_logs (
    id              BIGSERIAL       PRIMARY KEY,
    request_id      TEXT            NOT NULL,
    ts              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    model_version   TEXT            NOT NULL,
    predicted_class TEXT            NOT NULL,
    confidence      DOUBLE PRECISION NOT NULL,
    -- raw feature dict stored as JSONB for ad-hoc queries
    features        JSONB,
    -- optional ground-truth label joined later for accuracy tracking
    label           TEXT,
    latency_ms      DOUBLE PRECISION,
    CONSTRAINT uq_request_id UNIQUE (request_id)
);

CREATE INDEX IF NOT EXISTS idx_pred_logs_ts
    ON prediction_logs (ts DESC);
CREATE INDEX IF NOT EXISTS idx_pred_logs_model
    ON prediction_logs (model_version, ts DESC);
CREATE INDEX IF NOT EXISTS idx_pred_logs_class
    ON prediction_logs (predicted_class, ts DESC);

-- ── drift_reports ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS drift_reports (
    id                  BIGSERIAL       PRIMARY KEY,
    report_id           TEXT            NOT NULL UNIQUE,
    generated_at        TIMESTAMPTZ     NOT NULL,
    model_version       TEXT            NOT NULL,
    reference_size      INT             NOT NULL,
    current_size        INT             NOT NULL,
    window_start        TIMESTAMPTZ,
    window_end          TIMESTAMPTZ,
    overall_drifted     BOOLEAN         NOT NULL,
    drifted_features    TEXT[],
    drifted_count       INT             NOT NULL DEFAULT 0,
    critical_count      INT             NOT NULL DEFAULT 0,
    warning_count       INT             NOT NULL DEFAULT 0,
    drift_rate_pct      DOUBLE PRECISION,
    prediction_drifted  BOOLEAN         NOT NULL DEFAULT FALSE,
    prediction_psi      DOUBLE PRECISION,
    summary             JSONB
);

CREATE INDEX IF NOT EXISTS idx_drift_reports_ts
    ON drift_reports (generated_at DESC);
CREATE INDEX IF NOT EXISTS idx_drift_reports_model
    ON drift_reports (model_version, generated_at DESC);

-- ── drift_feature_results ─────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS drift_feature_results (
    id              BIGSERIAL       PRIMARY KEY,
    report_id       TEXT            NOT NULL REFERENCES drift_reports(report_id) ON DELETE CASCADE,
    feature         TEXT            NOT NULL,
    dtype           TEXT            NOT NULL,   -- numerical | categorical
    method          TEXT            NOT NULL,   -- ks | chi2
    statistic       DOUBLE PRECISION,
    p_value         DOUBLE PRECISION,
    psi             DOUBLE PRECISION,
    drifted         BOOLEAN         NOT NULL,
    severity        TEXT            NOT NULL,   -- none | warning | critical
    ref_mean        DOUBLE PRECISION,
    cur_mean        DOUBLE PRECISION,
    ref_std         DOUBLE PRECISION,
    cur_std         DOUBLE PRECISION
);

CREATE INDEX IF NOT EXISTS idx_drift_feat_report
    ON drift_feature_results (report_id);
CREATE INDEX IF NOT EXISTS idx_drift_feat_feature
    ON drift_feature_results (feature, drifted);

-- ── quality_reports ───────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS quality_reports (
    id              BIGSERIAL       PRIMARY KEY,
    report_id       TEXT            NOT NULL UNIQUE,
    generated_at    TIMESTAMPTZ     NOT NULL,
    model_version   TEXT            NOT NULL,
    window_size     INT             NOT NULL,
    window_start    TIMESTAMPTZ,
    window_end      TIMESTAMPTZ,
    overall_passed  BOOLEAN         NOT NULL,
    hard_failures   TEXT[],
    soft_warnings   TEXT[],
    hard_fail_count INT             NOT NULL DEFAULT 0,
    soft_warn_count INT             NOT NULL DEFAULT 0,
    summary         JSONB
);

CREATE INDEX IF NOT EXISTS idx_quality_reports_ts
    ON quality_reports (generated_at DESC);

-- ── quality_checks ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS quality_checks (
    id          BIGSERIAL   PRIMARY KEY,
    report_id   TEXT        NOT NULL REFERENCES quality_reports(report_id) ON DELETE CASCADE,
    check_name  TEXT        NOT NULL,
    severity    TEXT        NOT NULL,   -- hard | soft
    passed      BOOLEAN     NOT NULL,
    message     TEXT,
    details     JSONB
);

CREATE INDEX IF NOT EXISTS idx_quality_checks_report
    ON quality_checks (report_id, passed);

-- ── model_registry ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS model_registry (
    id                  BIGSERIAL       PRIMARY KEY,
    version_tag         TEXT            NOT NULL UNIQUE,
    registered_at       TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    status              TEXT            NOT NULL DEFAULT 'candidate',
    -- status: candidate | staging | production | retired | failed
    model_path          TEXT,
    pipeline_path       TEXT,
    eval_report_path    TEXT,
    val_accuracy        DOUBLE PRECISION,
    f1                  DOUBLE PRECISION,
    roc_auc             DOUBLE PRECISION,
    promoted_at         TIMESTAMPTZ,
    promoted_by         TEXT,           -- 'auto' | 'manual:<user>'
    retired_at          TIMESTAMPTZ,
    notes               TEXT,
    metrics             JSONB
);

CREATE INDEX IF NOT EXISTS idx_model_registry_status
    ON model_registry (status, registered_at DESC);

-- ── alert_log ─────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS alert_log (
    id          BIGSERIAL       PRIMARY KEY,
    ts          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    title       TEXT            NOT NULL,
    message     TEXT,
    severity    TEXT            NOT NULL,   -- info | warning | critical
    channel     TEXT            NOT NULL,
    sent        BOOLEAN         NOT NULL DEFAULT TRUE,
    labels      JSONB,
    environment TEXT
);

CREATE INDEX IF NOT EXISTS idx_alert_log_ts
    ON alert_log (ts DESC);
CREATE INDEX IF NOT EXISTS idx_alert_log_severity
    ON alert_log (severity, ts DESC);

-- ── Airflow metadata DB (separate DB; created by airflow db upgrade) ──────────
-- Nothing to create here — Airflow manages its own schema.
-- The POSTGRES_AIRFLOW_DB is set in .env and created by the postgres image
-- via POSTGRES_MULTIPLE_DATABASES if you add that extension, or manually:
-- (handled by docker-compose airflow-init command)

-- ── Seed: initial model_registry placeholder ──────────────────────────────────
-- Uncomment to pre-populate after your first training run.
-- INSERT INTO ml.model_registry (version_tag, status, notes)
-- VALUES ('baseline', 'retired', 'Initial seed — replace with real training run');
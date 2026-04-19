# ML Monitoring Pipeline

Production-grade ML monitoring stack: model training, inference API, feature drift detection, data quality checks, automated retraining, and full observability via Prometheus + Grafana.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Data Sources                             │
│              (DB / CSV / S3 via data/ingest.py)                 │
└───────────────────────────┬─────────────────────────────────────┘
                            │
              ┌─────────────▼──────────────┐
              │   Airflow: training_dag     │  weekly
              │  ingest→validate→features  │
              │  →train→evaluate→register  │
              └─────────────┬──────────────┘
                            │ artifacts/
                            ▼
┌───────────────────────────────────────────────────────────────┐
│                    FastAPI  :8000                              │
│   POST /predict   POST /predict/batch   GET /metrics          │
│   api/predict.py  api/logger.py  api/metrics.py               │
└───────────┬───────────────────────────────┬───────────────────┘
            │ prediction logs               │ /metrics
            ▼                               ▼
      Postgres :5432              Prometheus Exporter :9100
      prediction_logs             (drift + quality + API metrics)
      drift_reports                          │
      quality_reports                        ▼
      model_registry                  Prometheus :9090
      alert_log                              │
            │                               ▼
            └──────────────────────► Grafana :3000
                                     (dashboards + alerts)
            ┌──────────────────────────────────────┐
            │   Airflow: drift_check_dag  (6h)     │
            │   load_ref→fetch_live→drift→quality  │
            │   →push_metrics→gate                 │
            └──────────────┬───────────────────────┘
                           │ sets Airflow Variable
            ┌──────────────▼───────────────────────┐
            │  Airflow: retrain_trigger_dag (6h+15m)│
            │  check cooldown → trigger training   │
            └──────────────────────────────────────┘
```

---

## Quick Start

```bash
# 1. Clone and configure
git clone <repo>
cd ml_monitoring
cp .env.example .env
# Edit .env — fill in POSTGRES_PASSWORD, SLACK_WEBHOOK_URL, etc.

# 2. Start the stack
docker compose up -d

# 3. Verify services
docker compose ps
curl http://localhost:8000/health      # API
curl http://localhost:9090/-/healthy   # Prometheus
open http://localhost:3000             # Grafana (admin / see .env)
open http://localhost:8080             # Airflow  (admin / see .env)
```

### Generate Airflow Fernet key
```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
# Paste output into AIRFLOW_FERNET_KEY in .env
```

---

## Project Structure

```
ml_monitoring/
├── api/
│   ├── schemas.py          # FeatureSchema — single source of truth for columns/dtypes
│   ├── predict.py          # Model loading, hot-reload, predict() / predict_batch()
│   ├── logger.py           # Async prediction logger → Postgres + JSONL
│   ├── metrics.py          # In-process metrics registry (thread-safe singleton)
│   └── main.py             # FastAPI app: /predict /metrics /logs /model/reload
├── data/
│   ├── ingest.py           # Pull raw DataFrame from DB / CSV / S3
│   ├── validate.py         # 5 hard + 2 soft checks → ValidationReport
│   ├── features.py         # FeaturePipeline fit/transform/save/load
│   └── drift_injector.py   # 7 synthetic drift types for testing
├── training/
│   ├── train.py            # 7-step pipeline → model + pipeline artifacts
│   ├── evaluate.py         # Held-out eval: metrics, confusion matrix, calibration
│   └── register_model.py   # 3-gate promotion, registry JSON, symlinks, audit log
├── monitoring/
│   ├── drift_report.py     # KS / Chi2 / PSI per feature + prediction drift
│   ├── quality_report.py   # 7 quality checks → QualityReport
│   └── prometheus_exporter.py  # HTTP server exporting all metrics as Prometheus text
├── airflow/dags/
│   ├── training_dag.py         # Weekly full training pipeline
│   ├── drift_check_dag.py      # 6-hourly drift + quality check
│   └── retrain_trigger_dag.py  # Polls drift Variable, triggers training w/ cooldown
├── alerting/
│   ├── notify.py               # Slack / PagerDuty / email / webhook dispatcher
│   └── grafana_alerts.json     # 11 Grafana Unified Alerting rules
├── dockerfiles/
│   ├── Dockerfile.api          # Multi-stage FastAPI image
│   └── Dockerfile.airflow      # Extends official Airflow image
├── grafana/
│   ├── datasources.yaml        # Auto-provisioned Prometheus datasource
│   └── dashboards.json         # ML Monitoring dashboard (4 rows, 20+ panels)
├── prometheus/
│   └── prometheus.yml          # Scrape config: API, exporter, Airflow, node, cAdvisor
├── .github/workflows/
│   └── ci.yml                  # CI: lint + test on push
├── docker-compose.yml
├── .env.example
├── requirements.txt
├── init_db.sql                 # Postgres schema (6 tables)
└── README.md
```

---

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness — returns model version |
| `GET` | `/ready` | Readiness — 503 if model not loaded |
| `POST` | `/predict` | Single prediction |
| `POST` | `/predict/batch` | Batch predictions |
| `GET` | `/metrics` | Prometheus scrape endpoint |
| `GET` | `/metrics/summary` | Human-readable metrics snapshot |
| `POST` | `/model/reload` | Hot-reload model without restart |
| `GET` | `/logs` | Time-windowed prediction log query |

**Prediction request:**
```json
{
  "request_id": "req-123",
  "features": { "feature_1": 1.5, "feature_2": "cat_a" }
}
```

**Prediction response:**
```json
{
  "request_id": "req-123",
  "predicted_class": "1",
  "confidence": 0.87,
  "probabilities": { "0": 0.13, "1": 0.87 },
  "model_version": "20240101"
}
```

---

## Drift Detection

Runs every 6 hours via `drift_check_dag`. Compares the last `DRIFT_WINDOW_HOURS` of live predictions against the training reference dataset.

| Feature type | Test | Threshold |
|---|---|---|
| Numerical | Kolmogorov-Smirnov | p < `KS_P_THRESHOLD` (0.05) |
| Categorical | Chi-square | p < `CHI2_P_THRESHOLD` (0.05) |
| Both | PSI | ≥ 0.1 warning / ≥ 0.2 critical |
| Predictions | Chi-square on class dist | p < 0.05 |

If drift or quality failure is detected, `retrain_trigger_dag` fires `ml_training_pipeline` after a `RETRAIN_COOLDOWN_HOURS` (12h) guard.

---

## Grafana Alerts

| Alert | Severity | Condition |
|---|---|---|
| Critical Feature Drift | critical | `drift_critical_count > 0` for 5m |
| Feature Drift Warning | warning | `drift_warning_count > 0` for 15m |
| Prediction Distribution Drift | warning | `drift_prediction == 1` for 10m |
| Drift Report Stale | warning | report age > 12h |
| Data Quality Hard Failure | critical | `quality_hard_failures > 0` for 5m |
| High Feature Missing Rate | warning | missing % > 15% for 10m |
| Quality Report Stale | warning | report age > 12h |
| High API Error Rate | critical | error rate > 5% for 5m |
| High API Latency (p99) | warning | p99 > 2000ms for 5m |
| Low Prediction Confidence | warning | mean confidence < 0.55 for 15m |
| No Production Model | critical | model_loaded < 1 for 2m |

Routing: `critical` → Slack + PagerDuty. `warning` → Slack only.

---

## Development

### Local setup (no Docker)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env && source .env  # or use python-dotenv

# Run training pipeline
python -m training.train

# Start API
uvicorn api.main:app --reload --port 8000

# Run drift check manually
python -c "
from monitoring.drift_report import compute_drift_report
import pandas as pd
ref = pd.read_parquet('artifacts/features_<tag>.parquet')
cur = pd.read_parquet('artifacts/live_window.parquet')
report = compute_drift_report(ref, cur)
print(report.summary)
"

# Emit metrics once
python -m monitoring.prometheus_exporter --once
```

### Rebuild a single service
```bash
docker compose build ml_api
docker compose up -d --no-deps ml_api
```

### Tail logs
```bash
docker compose logs -f ml_api
docker compose logs -f airflow-scheduler
```

### Rollback model
```bash
# From inside the ml_api container or directly:
python -c "
from training.register_model import rollback_model
rollback_model(to_version='20231201')
"
# Then hot-reload the API:
curl -X POST http://localhost:8000/model/reload
```

---

## Environment Variables

See [`.env.example`](.env.example) for the full annotated list. Key variables:

| Variable | Default | Description |
|---|---|---|
| `ENVIRONMENT` | `production` | Label attached to all metrics and alerts |
| `AUTO_PROMOTE` | `false` | Auto-promote model after training |
| `DRIFT_WINDOW_HOURS` | `6` | Live window size for drift check |
| `RETRAIN_COOLDOWN_HOURS` | `12` | Minimum hours between auto-retrains |
| `SLACK_WEBHOOK_URL` | — | Required for Slack alerts |
| `PAGERDUTY_ROUTING_KEY` | — | Required for PagerDuty escalation |
| `AIRFLOW_FERNET_KEY` | — | **Required** — generate before first start |

---

## License

MIT
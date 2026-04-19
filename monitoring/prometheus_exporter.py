"""
monitoring/prometheus_exporter.py
Pulls metrics from three sources and writes Prometheus text-format:

  1. api/metrics.py   → in-process registry  (via to_prometheus_text())
  2. drift_report.py  → latest saved report   (PSI, drift booleans, counts)
  3. quality_report.py→ latest saved report   (missing rates, check pass/fail)

Runs as either:
  a) Standalone HTTP server  (default, port EXPORTER_PORT)
  b) Single-shot text dump   (python -m monitoring.prometheus_exporter --once)

Designed to be scraped by Prometheus (see prometheus/prometheus.yml).
"""

from __future__ import annotations

import argparse
import logging
import os

# import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Event, Thread
from typing import List, Optional

log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

EXPORTER_PORT = int(os.getenv("EXPORTER_PORT", "9100"))
SCRAPE_INTERVAL = int(os.getenv("SCRAPE_INTERVAL_SEC", "30"))
NAMESPACE = os.getenv("METRICS_NAMESPACE", "ml_monitor")


# ── Prometheus text helpers ───────────────────────────────────────────────────


def _gauge(
    name: str, value: float, labels: Optional[dict] = None, help_text: str = ""
) -> str:
    """Emit a single HELP+TYPE+metric line set."""
    full = f"{NAMESPACE}_{name}"
    label_str = ""
    if labels:
        pairs = ",".join(f'{k}="{v}"' for k, v in labels.items())
        label_str = f"{{{pairs}}}"
    lines = []
    if help_text:
        lines.append(f"# HELP {full} {help_text}")
    lines.append(f"# TYPE {full} gauge")
    lines.append(f"{full}{label_str} {value}")
    return "\n".join(lines)


def _counter(
    name: str, value: float, labels: Optional[dict] = None, help_text: str = ""
) -> str:
    full = f"{NAMESPACE}_{name}"
    label_str = ""
    if labels:
        pairs = ",".join(f'{k}="{v}"' for k, v in labels.items())
        label_str = f"{{{pairs}}}"
    lines = []
    if help_text:
        lines.append(f"# HELP {full} {help_text}")
    lines.append(f"# TYPE {full} counter")
    lines.append(f"{full}{label_str} {value}")
    return "\n".join(lines)


# ── Source 1: API in-process metrics ─────────────────────────────────────────


def _collect_api_metrics() -> str:
    """
    Tries to import the singleton metrics registry.
    Gracefully returns empty string if the API process is not running in-process.
    """
    try:
        from api.metrics import get_snapshot, to_prometheus_text

        return get_metrics_registry().to_prometheus_text()
    except Exception as e:
        log.debug("API metrics unavailable (expected if running standalone): %s", e)
        return ""


# ── Source 2: Drift report metrics ───────────────────────────────────────────


def _collect_drift_metrics() -> str:
    try:
        from monitoring.drift_report import load_latest_drift_report

        report = load_latest_drift_report()
        if report is None:
            log.debug("No drift report found.")
            return ""
    except Exception as e:
        log.error("Failed to load drift report: %s", e)
        return ""

    lines: List[str] = []

    # Overall drift flag
    lines.append(
        _gauge(
            "drift_overall",
            float(report.overall_drifted),
            labels={"model_version": report.model_version},
            help_text="1 if overall drift detected in the latest window, else 0.",
        )
    )

    # Drifted feature count
    lines.append(
        _gauge(
            "drift_feature_count",
            float(len(report.drifted_features)),
            labels={"model_version": report.model_version},
            help_text="Number of features with detected drift.",
        )
    )

    # Drift rate
    lines.append(
        _gauge(
            "drift_rate_pct",
            float(report.summary.get("drift_rate_pct", 0.0)),
            labels={"model_version": report.model_version},
            help_text="Percentage of checked features that drifted.",
        )
    )

    # Critical / warning counts
    lines.append(
        _gauge(
            "drift_critical_count",
            float(report.summary.get("critical_count", 0)),
            help_text="Features with critical drift severity.",
        )
    )
    lines.append(
        _gauge(
            "drift_warning_count",
            float(report.summary.get("warning_count", 0)),
            help_text="Features with warning drift severity.",
        )
    )

    # Per-feature PSI
    full_psi = f"{NAMESPACE}_drift_feature_psi"
    lines.append(f"# HELP {full_psi} PSI score per feature in the latest drift window.")
    lines.append(f"# TYPE {full_psi} gauge")
    for fr in report.feature_results:
        if fr.psi is not None:
            label_str = f'{{feature="{fr.feature}",severity="{fr.severity}"}}'
            lines.append(f"{full_psi}{label_str} {fr.psi}")

    # Per-feature drift flag
    full_flag = f"{NAMESPACE}_drift_feature_drifted"
    lines.append(f"# HELP {full_flag} 1 if feature drifted in the latest window.")
    lines.append(f"# TYPE {full_flag} gauge")
    for fr in report.feature_results:
        label_str = f'{{feature="{fr.feature}",method="{fr.method}"}}'
        lines.append(f"{full_flag}{label_str} {float(fr.drifted)}")

    # Prediction drift
    pd_ = report.prediction_drift
    lines.append(
        _gauge(
            "drift_prediction",
            float(pd_.drifted),
            labels={"model_version": report.model_version},
            help_text="1 if prediction distribution drift detected.",
        )
    )
    if pd_.psi is not None:
        lines.append(
            _gauge(
                "drift_prediction_psi",
                pd_.psi,
                help_text="PSI of prediction class distribution.",
            )
        )

    # Report age
    try:
        from datetime import datetime, timezone

        gen = datetime.fromisoformat(report.generated_at)
        age_s = (datetime.now(timezone.utc) - gen).total_seconds()
        lines.append(
            _gauge(
                "drift_report_age_seconds",
                age_s,
                help_text="Seconds since the latest drift report was generated.",
            )
        )
    except Exception:
        pass

    return "\n".join(lines)


# ── Source 3: Quality report metrics ─────────────────────────────────────────


def _collect_quality_metrics() -> str:
    try:
        from monitoring.quality_report import load_latest_quality_report

        report = load_latest_quality_report()
        if report is None:
            log.debug("No quality report found.")
            return ""
    except Exception as e:
        log.error("Failed to load quality report: %s", e)
        return ""

    lines: List[str] = []

    # Overall pass/fail
    lines.append(
        _gauge(
            "quality_overall_passed",
            float(report.overall_passed),
            labels={"model_version": report.model_version},
            help_text="1 if the latest quality report passed all hard checks.",
        )
    )

    lines.append(
        _gauge(
            "quality_hard_failures",
            float(len(report.hard_failures)),
            help_text="Number of hard check failures in the latest quality report.",
        )
    )
    lines.append(
        _gauge(
            "quality_soft_warnings",
            float(len(report.soft_warnings)),
            help_text="Number of soft warnings in the latest quality report.",
        )
    )
    lines.append(
        _gauge(
            "quality_window_size",
            float(report.window_size),
            help_text="Number of prediction records in the quality report window.",
        )
    )

    # Per-feature missing rate
    full_miss = f"{NAMESPACE}_quality_feature_missing_pct"
    lines.append(
        f"# HELP {full_miss} Missing value rate per feature in quality window."
    )
    lines.append(f"# TYPE {full_miss} gauge")
    for fq in report.feature_quality:
        label_str = f'{{feature="{fq.feature}"}}'
        lines.append(f"{full_miss}{label_str} {fq.missing_pct}")

    # Per-feature OOR rate (numerical)
    full_oor = f"{NAMESPACE}_quality_feature_oor_pct"
    lines.append(f"# HELP {full_oor} Out-of-range rate for numerical features.")
    lines.append(f"# TYPE {full_oor} gauge")
    for fq in report.feature_quality:
        if fq.oor_pct is not None:
            label_str = f'{{feature="{fq.feature}"}}'
            lines.append(f"{full_oor}{label_str} {fq.oor_pct}")

    # Per-feature unknown category rate
    full_unk = f"{NAMESPACE}_quality_feature_unknown_cat_pct"
    lines.append(f"# HELP {full_unk} Unknown category rate for categorical features.")
    lines.append(f"# TYPE {full_unk} gauge")
    for fq in report.feature_quality:
        if fq.unknown_cat_pct is not None:
            label_str = f'{{feature="{fq.feature}"}}'
            lines.append(f"{full_unk}{label_str} {fq.unknown_cat_pct}")

    # Per-check pass/fail
    full_chk = f"{NAMESPACE}_quality_check_passed"
    lines.append(f"# HELP {full_chk} 1 if the named quality check passed.")
    lines.append(f"# TYPE {full_chk} gauge")
    for chk in report.checks:
        label_str = f'{{check="{chk.check_name}",severity="{chk.severity}"}}'
        lines.append(f"{full_chk}{label_str} {float(chk.passed)}")

    # Report age
    try:
        from datetime import datetime, timezone

        gen = datetime.fromisoformat(report.generated_at)
        age_s = (datetime.now(timezone.utc) - gen).total_seconds()
        lines.append(
            _gauge(
                "quality_report_age_seconds",
                age_s,
                help_text="Seconds since the latest quality report was generated.",
            )
        )
    except Exception:
        pass

    return "\n".join(lines)


# ── Aggregator ────────────────────────────────────────────────────────────────


def collect_all_metrics() -> str:
    """Collect all metric sources and return Prometheus text payload."""
    sections = []
    for name, fn in [
        ("API", _collect_api_metrics),
        ("Drift", _collect_drift_metrics),
        ("Quality", _collect_quality_metrics),
    ]:
        try:
            text = fn()
            if text:
                sections.append(f"# ── {name} metrics ──\n" + text)
        except Exception as e:
            log.error("Collector '%s' crashed: %s", name, e)

    return "\n\n".join(sections) + "\n"


# ── HTTP server ───────────────────────────────────────────────────────────────


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        if self.path not in ("/metrics", "/"):
            self.send_response(404)
            self.end_headers()
            return
        try:
            payload = collect_all_metrics().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except Exception as e:
            log.error("Handler error: %s", e)
            self.send_response(500)
            self.end_headers()

    def log_message(self, fmt, *args):  # suppress default request logs
        log.debug(fmt, *args)


class PrometheusExporter:
    """Thread-safe exporter server. Can be embedded or run standalone."""

    def __init__(self, port: int = EXPORTER_PORT):
        self.port = port
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[Thread] = None
        self._stop = Event()

    def start(self) -> None:
        self._server = HTTPServer(("0.0.0.0", self.port), _Handler)
        self._thread = Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        log.info("Prometheus exporter listening on :%d/metrics", self.port)

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()
        self._stop.set()
        log.info("Prometheus exporter stopped.")


# ── CLI entry point ───────────────────────────────────────────────────────────


def _main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    )
    parser = argparse.ArgumentParser(description="ML Monitoring Prometheus Exporter")
    parser.add_argument(
        "--once", action="store_true", help="Print metrics once to stdout and exit."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=EXPORTER_PORT,
        help=f"HTTP port (default: {EXPORTER_PORT})",
    )
    args = parser.parse_args()

    if args.once:
        print(collect_all_metrics())
        return

    exporter = PrometheusExporter(port=args.port)
    exporter.start()
    try:
        while True:
            time.sleep(SCRAPE_INTERVAL)
    except KeyboardInterrupt:
        log.info("Interrupted, shutting down...")
        exporter.stop()


if __name__ == "__main__":
    _main()

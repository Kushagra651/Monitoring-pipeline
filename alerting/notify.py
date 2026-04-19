"""
alerting/notify.py
Multi-channel alert dispatcher used by all DAGs and monitoring tasks.

Supported channels (auto-selected or forced via `channel` arg):
  slack       — Incoming Webhook  (SLACK_WEBHOOK_URL)
  pagerduty   — Events API v2     (PAGERDUTY_ROUTING_KEY)
  email       — SMTP              (SMTP_HOST / SMTP_USER / SMTP_PASS / ALERT_EMAIL_TO)
  webhook     — Generic HTTP POST (ALERT_WEBHOOK_URL)
  log         — Python logger only (always active as fallback)

Usage
─────
  from alerting.notify import send_alert

  send_alert(
      channel="slack",           # or "pagerduty" | "email" | "webhook" | "all"
      title="Drift detected",
      message="Feature X drifted (PSI=0.23)",
      severity="warning",        # "info" | "warning" | "critical"
      labels={"dag": "ml_drift_check", "model_version": "20240101"},
  )

Dedup
─────
  Identical (title, severity) pairs are suppressed for DEDUP_WINDOW_SECONDS
  to prevent alert storms during repeated DAG retries.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import smtplib
import time
import urllib.request
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from threading import Lock
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

# ── Env config ────────────────────────────────────────────────────────────────

SLACK_WEBHOOK_URL      = os.getenv("SLACK_WEBHOOK_URL", "")
PAGERDUTY_ROUTING_KEY  = os.getenv("PAGERDUTY_ROUTING_KEY", "")
ALERT_WEBHOOK_URL      = os.getenv("ALERT_WEBHOOK_URL", "")
SMTP_HOST              = os.getenv("SMTP_HOST", "localhost")
SMTP_PORT              = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER              = os.getenv("SMTP_USER", "")
SMTP_PASS              = os.getenv("SMTP_PASS", "")
ALERT_EMAIL_FROM       = os.getenv("ALERT_EMAIL_FROM", "ml-monitor@company.com")
ALERT_EMAIL_TO         = os.getenv("ALERT_EMAIL_TO", "")   # comma-separated
DEDUP_WINDOW_SECONDS   = int(os.getenv("ALERT_DEDUP_WINDOW_SEC", "300"))   # 5 min
HTTP_TIMEOUT           = int(os.getenv("ALERT_HTTP_TIMEOUT_SEC", "10"))
ENV_LABEL              = os.getenv("ENVIRONMENT", "production")

# PagerDuty severity mapping
_PD_SEVERITY = {"info": "info", "warning": "warning", "critical": "critical"}
_SLACK_COLORS = {"info": "#36a64f", "warning": "#ffcc00", "critical": "#ff0000"}
_SLACK_EMOJI  = {"info": ":information_source:", "warning": ":warning:", "critical": ":fire:"}


# ── Alert record ──────────────────────────────────────────────────────────────

@dataclass
class Alert:
    title: str
    message: str
    severity: str = "warning"         # info | warning | critical
    channel: str = "slack"            # slack | pagerduty | email | webhook | all | log
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    environment: str = field(default_factory=lambda: ENV_LABEL)

    @property
    def dedup_key(self) -> str:
        raw = f"{self.title}:{self.severity}"
        return hashlib.md5(raw.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ── Dedup cache (in-process, thread-safe) ────────────────────────────────────

_dedup_cache: Dict[str, float] = {}
_dedup_lock = Lock()


def _is_duplicate(alert: Alert) -> bool:
    if DEDUP_WINDOW_SECONDS <= 0:
        return False
    key = alert.dedup_key
    now = time.time()
    with _dedup_lock:
        last_sent = _dedup_cache.get(key, 0)
        if now - last_sent < DEDUP_WINDOW_SECONDS:
            return True
        _dedup_cache[key] = now
        return False


# ── HTTP helper ───────────────────────────────────────────────────────────────

def _http_post(url: str, payload: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> int:
    """Simple urllib POST — no requests dependency required."""
    body = json.dumps(payload).encode("utf-8")
    hdrs = {"Content-Type": "application/json", **(headers or {})}
    req = urllib.request.Request(url, data=body, headers=hdrs, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            return resp.status
    except urllib.error.HTTPError as e:
        log.error("HTTP %d posting to %s: %s", e.code, url, e.read().decode(errors="replace"))
        return e.code
    except Exception as e:
        log.error("Request to %s failed: %s", url, e)
        return 0


# ── Channel implementations ───────────────────────────────────────────────────

def _send_slack(alert: Alert) -> bool:
    if not SLACK_WEBHOOK_URL:
        log.warning("SLACK_WEBHOOK_URL not set — skipping Slack alert.")
        return False

    label_text = "  |  ".join(f"{k}={v}" for k, v in alert.labels.items())
    payload = {
        "attachments": [{
            "color": _SLACK_COLORS.get(alert.severity, "#888888"),
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"{_SLACK_EMOJI.get(alert.severity, '')} {alert.title}",
                    },
                },
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": alert.message},
                },
                {
                    "type": "context",
                    "elements": [{"type": "mrkdwn", "text": (
                        f"*Severity:* {alert.severity.upper()}  |  "
                        f"*Env:* {alert.environment}  |  "
                        f"*Time:* {alert.timestamp}"
                        + (f"  |  {label_text}" if label_text else "")
                    )}],
                },
            ],
        }]
    }
    status = _http_post(SLACK_WEBHOOK_URL, payload)
    ok = status == 200
    if not ok:
        log.error("Slack alert failed (HTTP %d).", status)
    return ok


def _send_pagerduty(alert: Alert) -> bool:
    if not PAGERDUTY_ROUTING_KEY:
        log.warning("PAGERDUTY_ROUTING_KEY not set — skipping PagerDuty alert.")
        return False

    payload = {
        "routing_key": PAGERDUTY_ROUTING_KEY,
        "event_action": "trigger",
        "dedup_key": alert.dedup_key,
        "payload": {
            "summary": alert.title,
            "severity": _PD_SEVERITY.get(alert.severity, "warning"),
            "source": f"ml-monitor/{alert.environment}",
            "timestamp": alert.timestamp,
            "custom_details": {
                "message": alert.message,
                "labels": alert.labels,
            },
        },
    }
    status = _http_post(
        "https://events.pagerduty.com/v2/enqueue",
        payload,
        headers={"X-Routing-Key": PAGERDUTY_ROUTING_KEY},
    )
    ok = status in (200, 202)
    if not ok:
        log.error("PagerDuty alert failed (HTTP %d).", status)
    return ok


def _send_email(alert: Alert) -> bool:
    if not ALERT_EMAIL_TO or not SMTP_HOST:
        log.warning("SMTP not configured — skipping email alert.")
        return False

    recipients = [r.strip() for r in ALERT_EMAIL_TO.split(",") if r.strip()]
    label_html = "".join(
        f"<tr><td><b>{k}</b></td><td>{v}</td></tr>" for k, v in alert.labels.items()
    )
    body_html = f"""
    <html><body>
      <h2 style="color:{'red' if alert.severity=='critical' else 'orange'}">
        {alert.title}
      </h2>
      <p><b>Severity:</b> {alert.severity.upper()} | <b>Env:</b> {alert.environment}</p>
      <pre style="background:#f4f4f4;padding:10px">{alert.message}</pre>
      <table border="1" cellpadding="4">{label_html}</table>
      <p><small>{alert.timestamp}</small></p>
    </body></html>
    """

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"[ML Monitor] [{alert.severity.upper()}] {alert.title}"
    msg["From"]    = ALERT_EMAIL_FROM
    msg["To"]      = ", ".join(recipients)
    msg.attach(MIMEText(alert.message, "plain"))
    msg.attach(MIMEText(body_html, "html"))

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=HTTP_TIMEOUT) as s:
            s.ehlo()
            if SMTP_USER:
                s.starttls()
                s.login(SMTP_USER, SMTP_PASS)
            s.sendmail(ALERT_EMAIL_FROM, recipients, msg.as_string())
        return True
    except Exception as e:
        log.error("Email alert failed: %s", e)
        return False


def _send_webhook(alert: Alert) -> bool:
    if not ALERT_WEBHOOK_URL:
        log.warning("ALERT_WEBHOOK_URL not set — skipping webhook alert.")
        return False
    status = _http_post(ALERT_WEBHOOK_URL, alert.to_dict())
    ok = 200 <= status < 300
    if not ok:
        log.error("Webhook alert failed (HTTP %d).", status)
    return ok


# ── Dispatcher ────────────────────────────────────────────────────────────────

_CHANNEL_FN = {
    "slack":      _send_slack,
    "pagerduty":  _send_pagerduty,
    "email":      _send_email,
    "webhook":    _send_webhook,
}


def send_alert(
    title: str,
    message: str,
    severity: str = "warning",
    channel: str = "slack",
    labels: Optional[Dict[str, str]] = None,
    # legacy kwarg aliases used in older DAG versions
    level: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
) -> Dict[str, bool]:
    """
    Dispatch an alert to one or more channels.

    Returns dict of {channel: success_bool} for every channel attempted.

    Parameters
    ──────────
    title    : Short alert headline
    message  : Full description / stack trace / metric values
    severity : "info" | "warning" | "critical"
    channel  : "slack" | "pagerduty" | "email" | "webhook" | "all" | "log"
    labels   : Key-value metadata attached to the alert
    """
    # Normalise legacy kwargs
    severity = severity or level or "warning"
    labels   = labels or tags or {}

    alert = Alert(
        title=title,
        message=message,
        severity=severity,
        channel=channel,
        labels={**labels, "environment": ENV_LABEL},
    )

    # Always log
    log.info(
        "ALERT [%s] %s | %s | labels=%s",
        alert.severity.upper(), alert.title, alert.message[:120], alert.labels,
    )

    if _is_duplicate(alert):
        log.info("Alert suppressed (dedup window %ds): %s", DEDUP_WINDOW_SECONDS, alert.title)
        return {"dedup": False}

    if channel == "log":
        return {"log": True}

    channels: List[str] = list(_CHANNEL_FN.keys()) if channel == "all" else [channel]
    results: Dict[str, bool] = {}

    for ch in channels:
        fn = _CHANNEL_FN.get(ch)
        if fn is None:
            log.warning("Unknown alert channel: %s", ch)
            results[ch] = False
            continue
        try:
            results[ch] = fn(alert)
        except Exception as e:
            log.error("Channel '%s' raised: %s", ch, e)
            results[ch] = False

    # If ALL configured channels failed → escalate to log at ERROR level
    if results and not any(results.values()):
        log.error(
            "ALL alert channels failed for: [%s] %s",
            alert.severity.upper(), alert.title,
        )

    return results


# ── Convenience wrappers ──────────────────────────────────────────────────────

def alert_info(title: str, message: str, **kwargs) -> Dict[str, bool]:
    return send_alert(title=title, message=message, severity="info", **kwargs)


def alert_warning(title: str, message: str, **kwargs) -> Dict[str, bool]:
    return send_alert(title=title, message=message, severity="warning", **kwargs)


def alert_critical(title: str, message: str, **kwargs) -> Dict[str, bool]:
    return send_alert(title=title, message=message, severity="critical", **kwargs)
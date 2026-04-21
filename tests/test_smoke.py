# tests/test_smoke.py
def test_schemas_import():
    from api.schemas import PredictionInput, PredictionOutput, HealthResponse
    assert PredictionInput is not None

def test_predict_imports():
    from api.predict import get_model_info, PredictionResult
    assert get_model_info is not None

def test_notify_imports():
    from alerting.notify import send_alert, alert_warning
    assert send_alert is not None

def test_drift_report_imports():
    from monitoring.drift_report import compute_drift_report, DriftReport
    assert DriftReport is not None

def test_quality_report_imports():
    from monitoring.quality_report import compute_quality_report, QualityReport
    assert QualityReport is not None
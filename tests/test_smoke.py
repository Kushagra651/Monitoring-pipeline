# tests/test_smoke.py


def test_schemas_import():
    from api.schemas import PredictionInput

    assert PredictionInput is not None


def test_predict_imports():
    from api.predict import get_model_info

    assert get_model_info is not None


def test_notify_imports():
    from alerting.notify import send_alert

    assert send_alert is not None


def test_drift_report_imports():
    from monitoring.drift_report import DriftReport

    assert DriftReport is not None


def test_quality_report_imports():
    from monitoring.quality_report import QualityReport

    assert QualityReport is not None
from src.evaluate_model import TrainingMetrics, calculate_metrics


def test_training_metrics_dataclass():
    metrics = TrainingMetrics(accuracy=0.95, precision=0.94, recall=0.93, f1=0.935)
    assert metrics.accuracy == 0.95
    assert metrics.precision == 0.94
    assert metrics.recall == 0.93
    assert metrics.f1 == 0.935
    assert metrics.to_dict() == {
        "accuracy": 0.95,
        "precision": 0.94,
        "recall": 0.93,
        "f1": 0.935,
    }


def test_calculate_metrics():
    y_true = [0, 1, 1, 0]
    y_pred = [0, 1, 0, 0]
    metrics = calculate_metrics(y_true, y_pred)
    assert isinstance(metrics, TrainingMetrics)
    assert 0.0 <= metrics.accuracy <= 1.0

from src.predict import PredictionResult

def test_prediction_result():
    res = PredictionResult(label="normal", confidence=0.98, elapsed_ms=12.34)
    assert res.label == "normal"
    assert res.confidence == 0.98
    assert res.elapsed_ms == 12.34
    assert res.to_dict() == {"label": "normal", "confidence": 0.98, "elapsed_ms": 12.34}

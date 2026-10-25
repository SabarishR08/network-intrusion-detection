import json
from pathlib import Path
from src.utils import save_model_metadata


def test_save_model_metadata(tmp_path: Path) -> None:
    output_file = tmp_path / "metadata.json"
    res_path = save_model_metadata("RandomForestClassifier", 0.985, output_path=output_file)
    assert res_path.exists()

    with open(res_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["model_name"] == "RandomForestClassifier"
    assert data["accuracy"] == 0.985
    assert "date" in data

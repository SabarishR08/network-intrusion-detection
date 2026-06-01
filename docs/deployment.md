# Deployment Instructions

## Windows Setup

1. Open the project folder in VS Code.
2. Run `setup.bat` to install the Python dependencies.
3. Run `run.bat` to train the model and generate the presentation.

## Manual Commands

```bat
python -m pip install -r requirements.txt
python -m src.train_model
python build_presentation.py
streamlit run app/streamlit_app.py
```

## Output Files

- `models/model.pkl`
- `models/encoders.pkl`
- `models/scaler.pkl`
- `reports/metrics.json`
- `reports/figures/*.png`
- `reports/predictions.csv`

## Notes

- The training script automatically extracts the dataset archive when the CSV files are missing.
- The dashboard expects the model artifacts produced by the training step.

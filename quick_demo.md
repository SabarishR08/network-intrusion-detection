# Quick Demo Guide

1. Install dependencies:

```powershell
py -3 -m pip install -r requirements.txt
```

2. Train the model:

```powershell
py -3 -m src.train_model
```

3. Launch the Streamlit dashboard:

```powershell
py -3 -m streamlit run app/streamlit_app.py
```

4. Upload a sample CSV:
- `demo/sample_input.csv`

5. Check the outputs:
- Predictions export: `demo/sample_predictions.csv`
- Figures: `screenshots/confusion_matrix.png`, `screenshots/roc_curve.png`, `screenshots/feature_importance.png`

6. If the app says a model is missing, rerun training so `models/model.pkl` is regenerated.

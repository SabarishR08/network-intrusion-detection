# network-intrusion-detection â€” daily improvement backlog
# ML-based network intrusion detection system (Python, Streamlit, scikit-learn)

- [x] Add module-level docstrings to src/data_preprocessing.py
- [x] Add module-level docstrings to src/feature_engineering.py
- [x] Add type hints to all functions in src/utils.py
- [x] Add type hints to all functions in src/predict.py
- [x] Add a __version__ = "0.1.0" to src/__init__.py
- [x] Create a smoke-test CI workflow at .github/workflows/ci.yml that runs python -m compileall on src/, scripts/, and main.py to catch syntax errors on every push and pull_request (no dependencies needed)
- [ ] Add logging.getLogger(__name__) to src/train_model.py replacing bare print() calls
- [x] Add logging.getLogger(__name__) to src/evaluate_model.py replacing bare print() calls
- [ ] Add a constants.py in src/ for feature column names and label mappings
- [x] Add input validation in src/predict.py: check feature vector length before inference
- [x] Add a try/except around model.predict() in src/predict.py with a clear error message
- [ ] Add a CONTRIBUTING.md at repo root with setup steps and dataset download instructions
- [ ] Add a .editorconfig at repo root (4-space indent, utf-8, lf)
- [ ] Add a requirements-dev.txt with pytest and ruff
- [ ] Add a pytest.ini at repo root with testpaths = ["tests"]
- [ ] Create tests/test_smoke.py that imports src and asserts the module loads without error
- [ ] Add a .gitattributes normalising line endings
- [ ] Add __all__ to src/__init__.py listing public symbols
- [ ] Add elapsed-time logging around the model training loop
- [ ] Add elapsed-time logging around the predict() call
- [ ] Add a retry wrapper for any file I/O that could fail on missing dataset
- [ ] Add a validate_dataframe(df, expected_columns) helper in src/utils.py
- [ ] Add a normalise_labels(y) helper that maps raw label strings to int class indices
- [ ] Add a confusion_matrix summary log after model evaluation
- [ ] Add a save_model_metadata() function that writes model name, date, and accuracy to JSON
- [ ] Add a load_model_with_fallback() that logs a warning if model file is missing
- [ ] Add a dataclass PredictionResult(label, confidence, elapsed_ms) replacing plain dict
- [ ] Add a dataclass TrainingMetrics(accuracy, precision, recall, f1) for evaluation output
- [ ] Add a CLI --evaluate flag to main.py that runs evaluation and prints metrics
- [ ] Add a CLI --version flag to main.py that prints the version from src/__init__.py
- [ ] Add a brief Architecture section to README.md describing the preprocessing pipeline


<!-- backlog top-up 2026-09-19 (file-verified) -->
- [ ] Add logging.getLogger(__name__) to src/data_preprocessing.py replacing bare print() calls
- [ ] Add logging.getLogger(__name__) to src/feature_engineering.py replacing bare print() calls
- [ ] Add logging.getLogger(__name__) to src/predict.py replacing bare print() calls
- [ ] Add a module-level docstring to src/utils.py describing the shared helpers
- [ ] Add type hints to all function signatures in src/utils.py
- [ ] Add type hints to all function signatures in src/feature_engineering.py
- [ ] Add a module-level docstring to scripts/create_demo.py describing what the demo script generates
- [ ] Add a module-level docstring to app/streamlit_app.py describing the dashboard layout


<!-- backlog top-up 2026-09-19 batch 2 (file+verb deduped) -->
- [ ] Add a module-level docstring to src/evaluate_model.py describing the evaluation metrics
- [ ] Add a module-level docstring to src/train_model.py describing the training pipeline
- [ ] Add a module-level docstring to src/predict.py describing the prediction entry point
- [ ] Add a module-level docstring to src/data_preprocessing.py describing the preprocessing steps
- [ ] Add type hints to all function signatures in src/data_preprocessing.py
- [ ] Add type hints to all function signatures in src/evaluate_model.py
- [ ] Add type hints to all function signatures in src/train_model.py
- [ ] Add type hints to all function signatures in src/predict.py
- [ ] Add a module-level docstring to main.py describing the command-line entry point
- [ ] Add type hints to all function signatures in scripts/create_demo.py
- [ ] Add a module-level docstring to build_presentation.py describing the presentation builder
- [ ] Add a module-level docstring to src/feature_engineering.py describing derived features
- [ ] Add type hints to all function signatures in main.py


<!-- backlog top-up 2026-09-19 batch 2b (file+verb deduped) -->
- [ ] Add a module-level docstring to src/__init__.py describing the package

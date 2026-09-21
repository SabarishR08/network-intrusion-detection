# Contributing to ML Project

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-org/your-repo.git
   cd your-repo
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:
   ```bash
   streamlit run app/streamlit_app.py
   ```

## Dataset

The dataset used for training and evaluation is not included in the repository due to size constraints. To obtain it:

1. Visit [Dataset Source URL] and download the raw data file.
2. Place the downloaded file in `data/raw/` directory.
3. Run the preprocessing script to generate processed data:
   ```bash
   python src/data_preprocessing.py
   ```

Alternatively, you can generate a synthetic dataset for testing:
```bash
python scripts/create_demo.py
```

## Adding New Features

1. Create a new branch from `main`: `git checkout -b feature/your-feature`
2. Implement your changes following the existing code style.
3. Add or update tests as needed.
4. Run `pytest` to verify your changes don't break existing functionality.
5. Commit with a descriptive message and open a pull request.

## Guidelines
- Follow PEP 8 for Python code.
- Keep changes focused and small.
- Update documentation if user-facing behavior changes.
- Ensure all new dependencies are added to `requirements.txt`.

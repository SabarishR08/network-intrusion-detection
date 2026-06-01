# Methodology

## Data Preparation

The dataset contains 41 connection-level features with categorical and numerical values. The training script normalizes the target labels, imputes missing values, applies one-hot encoding to categorical fields, and scales numeric features for models that benefit from normalized input.

## Modeling Strategy

Three baseline models are trained on the same preprocessing pipeline:

- Random Forest
- Gradient Boosting
- Logistic Regression

An optional XGBoost model can be enabled if the library is available.

## Evaluation

The project reports:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

The best model is selected using validation F1-score.

## Inference

The provided Test CSV is unlabeled. The pipeline therefore performs inference-only scoring on that file and exports predicted labels plus probabilities to a CSV file.

## Cybersecurity Relevance

The project is directly relevant to intrusion detection operations because it turns connection records into actionable anomaly predictions, which is the same basic requirement found in network security monitoring and SOC triage.

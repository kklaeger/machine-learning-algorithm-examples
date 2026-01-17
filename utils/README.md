# Utils

This folder contains small shared helper modules used across multiple examples in this repository. The goal is to keep
the algorithm implementations focused on the core learning objectives while ensuring consistent data handling and
evaluation.

## Modules

### [Data Utils](data_utils.py)

Utilities for loading and splitting tabular CSV datasets.

- `load_training_data(file_path)`  
  Loads a CSV file and returns feature matrix `X` and target vector `y` (last column is assumed to be the target).

- `split_data(X, y, test_ratio=0.2, seed=42, shuffle=True)`  
  Deterministic train/test split for reproducible experiments. Returns `X_train, y_train, X_test, y_test`.

### [Metrics](metrics.py)

Evaluation metrics for regression and classification tasks.

- `compute_mse_rmse(y_true, y_pred)`  
  Mean Squared Error and Root Mean Squared Error (regression).

- `compute_cross_entropy_loss(y_true, y_pred)`  
  Binary cross-entropy (log loss) for probabilistic binary classifiers.

- `compute_accuracy(y_true, y_pred, threshold=0.5)`  
  Accuracy for:
    - binary class labels
    - binary probabilities (using `threshold`)
    - multiclass probabilities (using `argmax`)

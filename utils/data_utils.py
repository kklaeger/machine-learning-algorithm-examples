import numpy as np
from pathlib import Path

DEFAULT_SEED = 42

# Ensure reproducible results
np.random.seed(DEFAULT_SEED)


def load_data(file_path, with_target=True):
    """
    Loads training data from a CSV file and split it into features and target values.

    Parameters:
        file_path (str | Path): Path to the CSV training data file.
        with_target (bool):     Whether the last column is the target variable.

    Returns:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): True target values (if with_target is True).
    """
    data = np.loadtxt(fname=file_path, delimiter=',', skiprows=1, dtype=float)
    if with_target:
        X = data[:, :-1]
        y = data[:, -1]
        return X, y
    else:
        X = data
        return X


def split_data(X, y, test_ratio=0.2, seed=DEFAULT_SEED, shuffle=True):
    """
    Splits the dataset into training and testing sets.

    Parameters:
        X (np.ndarray):     Feature matrix.
        y (np.ndarray):     True target values.
        test_ratio (float): Proportion of the dataset to include in the test split.
        seed (int):         Random seed for reproducibility.
        shuffle (bool):     Whether to shuffle the data before splitting.

    Returns:
        X_train (np.ndarray):   Training feature matrix.
        y_train (np.ndarray):   Training target values.
        X_test (np.ndarray):    Testing feature matrix.
        y_test (np.ndarray):    Testing target values.
    """
    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio must be between 0 and 1")

    n = len(X)
    if n != len(y):
        raise ValueError("Features and labels must have the same number of samples")

    # Generate indices and shuffle if required
    indices = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    # Determine the number of test samples and ensure at least one sample in test set if test_ratio > 0
    test_count = int(round(n * test_ratio))
    test_count = max(1, min(test_count, n - 1))

    # if user explicitly wants no test set
    if test_ratio == 0:
        test_count = 0

    test_indices = indices[:test_count]
    train_indices = indices[test_count:]

    return X[train_indices], y[train_indices], X[test_indices], y[test_indices]

import numpy as np
from pathlib import Path

from utils.data_utils import load_data

SEED = 42

# Ensure reproducible results
np.random.seed(SEED)

np.set_printoptions(precision=2, suppress=True)

BASE_DIR = Path(__file__).resolve().parent
TRAINING_DATA_PATH = BASE_DIR / "data" / "training_data.csv"
CROSS_VAL_DATA_PATH = BASE_DIR / "data" / "cross_val_data.csv"


def scale_features(X, mu=None, sigma=None):
    """
    Scales the features using mean and standard deviation.

    If mu and sigma are None, they are computed from X and returned.
    If mu and sigma are provided, only the scaled features are returned.

    Parameters:
        X (np.ndarray):     Feature matrix.
        mu (np.ndarray):    Mean values for each feature. If None, compute from X.
        sigma (np.ndarray): Standard deviation for each feature. If None, compute from X.

    Returns:
        X_scaled (np.ndarray):  Scaled feature matrix.
        mu (np.ndarray):        Mean values used for scaling.
        sigma (np.ndarray):     Standard deviation values used for scaling.
    """
    compute_params = (mu is None) or (sigma is None)

    if compute_params:
        mu = np.mean(X, axis=0)
        sigma = np.std(X, axis=0)

    # Avoid division by zero for constant features
    sigma = np.where(sigma == 0, 1.0, sigma)
    X_scaled = (X - mu) / sigma

    if compute_params:
        return X_scaled, mu, sigma
    else:
        return X_scaled


def estimate_gaussian(X):
    """
    Estimates the parameters of a multivariate Gaussian distribution.

    Parameters:
        X (np.ndarray): Feature matrix of shape (m, n)

    Returns:
        mu (np.ndarray):    Mean vector of shape (n,)
        sigma (np.ndarray): Covariance matrix of shape (n, n)
    """
    mu = np.mean(X, axis=0)
    sigma = np.cov(X, rowvar=False)

    return mu, sigma


def multivariate_gaussian(X, mu, sigma):
    """
    Computes the probability density of each example in X under a multivariate Gaussian distribution.

    Parameters:
        X (np.ndarray): Feature matrix of shape (m, n)
        mu (np.ndarray): Mean vector of shape (n,)
        sigma (np.ndarray): Covariance matrix of shape (n, n)

    Returns:
        p (np.ndarray): Probability densities of shape (m,)
    """
    n = mu.shape[0]
    X_mu = X - mu

    sigma_inv = np.linalg.inv(sigma)
    det_sigma = np.linalg.det(sigma)

    coeff = 1 / np.sqrt((2 * np.pi) ** n * det_sigma)
    exp_term = np.exp(-0.5 * np.sum((X_mu @ sigma_inv) * X_mu, axis=1))

    return coeff * exp_term


def select_threshold(y_val, p_val):
    """
    Finds the best threshold (epsilon) to identify anomalies based on the F1 score.

    Parameters:
        y_val (np.ndarray): Ground truth labels (0 = normal, 1 = anomaly)
        p_val (np.ndarray): Probability values for validation set

    Returns:
        best_epsilon (float):   Selected threshold
        best_f1 (float):        Best F1 score achieved
    """
    best_epsilon = 0
    best_f1 = 0

    epsilons = np.linspace(p_val.min(), p_val.max(), 1000)

    for epsilon in epsilons:
        y_pred = (p_val < epsilon).astype(int)

        tp = int(np.sum((y_pred == 1) & (y_val == 1)))
        fp = int(np.sum((y_pred == 1) & (y_val == 0)))
        fn = int(np.sum((y_pred == 0) & (y_val == 1)))

        if tp + fp == 0 or tp + fn == 0:
            continue

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)

        if f1 > best_f1:
            best_f1 = f1
            best_epsilon = epsilon

    return best_epsilon, best_f1


def predict(p, epsilon):
    """
    Predicts whether examples are anomalies based on probability values.

    Parameters:
        p (np.ndarray):       Probability values p(x)
        epsilon (float):      Threshold value

    Returns:
        y_pred (np.ndarray):  Predictions (1 = anomaly, 0 = normal)
    """
    return (p < epsilon).astype(int)


def main():
    # Load the training and cross validation data from the CSV files
    X_train = load_data(TRAINING_DATA_PATH, with_target=False)
    X_val, y_val = load_data(CROSS_VAL_DATA_PATH, with_target=True)

    # Scale features
    X_train_scaled, mu_scale, sigma_scale = scale_features(X_train)
    X_val_scaled = scale_features(X_val, mu_scale, sigma_scale)

    # Estimate Gaussian parameters
    mu, sigma = estimate_gaussian(X_train_scaled)

    # Compute probabilities
    p_train = multivariate_gaussian(X_train_scaled, mu, sigma)
    p_val = multivariate_gaussian(X_val_scaled, mu, sigma)

    # Select threshold using cross-validation set
    epsilon, best_f1 = select_threshold(y_val, p_val)

    y_val_pred = predict(p_val, epsilon)

    print("\nEvaluation:")
    print(f"Selected epsilon: {epsilon:.3e}")
    print(f"Best F1 score:    {best_f1:.3f}")
    print("Predicted anomalies:", int(np.sum(y_val_pred)))
    print("True anomalies:     ", int(np.sum(y_val)))

    # Predict anomalies for test transactions
    test_transactions = np.array([
        [25.0, 14, 2, 1.0],  # normal purchase
        [950.0, 3, 12, 800.0],  # suspicious transaction
        [45.0, 16, 1, 2.5],  # normal purchase
    ], dtype=float)

    # Scale test transactions
    test_transactions_scaled = scale_features(test_transactions, mu_scale, sigma_scale)

    # Compute probabilities
    p_test = multivariate_gaussian(test_transactions_scaled, mu, sigma)

    # Predict anomalies
    predictions = predict(p_test, epsilon)
    print("\nPredictions for test transactions:")
    print(f"Transaction 1 -> anomaly={predictions[0]}")  # 0 = normal
    print(f"Transaction 2 -> anomaly={predictions[1]}")  # 1 = anomaly
    print(f"Transaction 3 -> anomaly={predictions[2]}")  # 0 = normal


if __name__ == "__main__":
    main()

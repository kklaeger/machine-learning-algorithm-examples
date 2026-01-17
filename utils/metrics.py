import numpy as np


# --- Regression Metrics ---
def compute_mse_rmse(y_true, y_pred):
    """
    Computes the Mean Squared Error (MSE) and Root Mean Squared Error (RMSE)
    between true and predicted target values.

    Parameters:
        y_true (np.ndarray): True target values.
        y_pred (np.ndarray): Predicted target values.
    Returns:
        mse (float):    Mean Squared Error.
        rmse (float):   Root Mean Squared Error.
    """
    errors = y_pred - y_true
    mse = np.mean(errors ** 2)
    rmse = np.sqrt(mse)

    return mse, rmse


# --- Classification Metrics ---
def compute_cross_entropy_loss(y_true, y_pred):
    """
    Computes the cross-entropy loss for logistic regression.

    Parameters:
        y_true (np.ndarray): True target values (0 or 1).
        y_pred (np.ndarray): Predicted probabilities in the range (0, 1).
    Returns:
        loss (np.ndarray): Cross-entropy loss for each sample.
    """
    # Avoid log(0) by clipping probabilities
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)

    loss = -y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)
    return np.mean(loss)


def compute_accuracy(y_true, y_pred, threshold=0.5):
    """
    Computes the accuracy for binary and multi-class classification. Accepts both class labels and predicted
    probabilities.

    Parameters:
        y_true (np.ndarray): True target values.
        y_pred (np.ndarray): Predicted target values.
        threshold (float):   Threshold to convert predicted probabilities to class labels.
    Returns:
        accuracy (float):    Accuracy score.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Case 1: multiclass probabilities -> argmax
    if y_pred.ndim == 2:
        y_hat = np.argmax(y_pred, axis=1)

    # Case 2: binary class labels (0/1)
    elif set(np.unique(y_pred)).issubset({0, 1}):
        y_hat = y_pred.astype(int)

    # Case 3: binary probabilities
    else:
        y_hat = (y_pred >= threshold).astype(int)

    return np.mean(y_hat == y_true)

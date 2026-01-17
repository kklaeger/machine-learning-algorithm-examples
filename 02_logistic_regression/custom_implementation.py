import numpy as np
import copy
from pathlib import Path

from utils.data_utils import split_data, load_data
from utils.metrics import compute_cross_entropy_loss, compute_accuracy

SEED = 42

# Ensure reproducible results
np.random.seed(SEED)

np.set_printoptions(precision=2, suppress=True)

BASE_DIR = Path(__file__).resolve().parent
TRAINING_DATA_PATH = BASE_DIR / "data" / "training_data.csv"


def sigmoid(z):
    """
    Apply the sigmoid function.

    Parameters:
        z (np.ndarray | float): Input value(s).

    Returns:
        sigmoid (np.ndarray | float): Sigmoid-transformed output in the range [0, 1].
    """
    return 1 / (1 + np.exp(-z))


def model_output(X, w, b):
    """
    Computes the raw output of the model f(x) = sigmoid(w·x + b).

    Parameters:
        X (np.ndarray): Feature matrix.
        w (np.ndarray): Weight vector.
        b (float):      Bias term.

    Returns:
        y_hat (np.ndarray): Predicted values for each input sample.
    """
    return sigmoid(X @ w + b)


def compute_loss(y_hat, y):
    """
    Compute the cross-entropy loss for individual predictions.

    Parameters:
        y_hat (np.ndarray): Predicted probabilities in the range (0, 1).
        y (np.ndarray): True target values (0 or 1).

    Returns:
        loss (np.ndarray): Cross-entropy loss for each sample.
    """
    # Avoid log(0) by clipping probabilities
    eps = 1e-15
    y_hat = np.clip(y_hat, eps, 1 - eps)

    loss = -y * np.log(y_hat) - (1 - y) * np.log(1 - y_hat)

    return loss


def compute_cost(X, y, w, b):
    """
    Computes the cost for logistic regression (cross-entropy / log loss).

    Parameters:
        X (np.ndarray):     Feature matrix including.
        y (np.ndarray):     True target values.
        w (np.ndarray):     Weight vector.
        b (float):          Bias term.

    Returns:
        cost (float): The cost value.
    """
    y_hat = model_output(X, w, b)

    losses = compute_loss(y_hat, y)
    cost = np.mean(losses)

    return cost


def compute_gradients(X, y, w, b):
    """
    Computes the gradient of the MSE cost function for linear regression.

    Parameters:
        X (np.ndarray):     Feature matrix including bias term.
        y (np.ndarray):     True target values.
        w (np.ndarray):     Weight vector.
        b (float):          Bias term.

    Returns:
        dw (np.ndarray): Weight gradients.
        db (float):      Bias gradient.
    """
    m = len(y)

    y_hat = model_output(X, w, b)
    errors = y_hat - y

    dw = (1 / m) * (X.T @ errors)
    db = (1 / m) * np.sum(errors)

    return dw, db


def gradient_descent(X, y, w_init, b_init, alpha, iterations):
    """
    Performs batch gradient descent to learn weights and bias.

    Parameters:
        X (np.ndarray):         Feature matrix including bias term.
        y (np.ndarray):         True target values.
        w_init (np.ndarray):    Initial weight vector.
        b_init (float):         Initial bias term.
        alpha (float):          Learning rate.
        iterations (int):       Number of iterations.

    Returns:
        w (np.ndarray): Learned weights.
        b (float):      Learned bias.
    """
    print_every = 1000
    w = copy.deepcopy(w_init)
    b = b_init

    for i in range(iterations):
        dw, db = compute_gradients(X, y, w, b)

        w = w - alpha * dw
        b = b - alpha * db

        cost = compute_cost(X, y, w, b)

        if i % print_every == 0:
            print(f"Iteration {i:4d} | Cost: {cost:.6e}")

    return w, b


def scale_features(X):
    """
    Scales the features using mean and standard deviation.

    Parameters:
        X (np.ndarray): Feature matrix.

    Returns:
        X_scaled (np.ndarray):  Scaled feature matrix.
        mu (np.ndarray):        Mean value of each feature.
        sigma (np.ndarray):     Standard deviation of each feature.
    """
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)

    X_scaled = (X - mu) / sigma

    return X_scaled, mu, sigma


def predict_scaled_proba(X, w, b, mu, sigma):
    """
    Predict the probabilities using learned weights and bias.

        Parameters:
        X (np.ndarray):     Feature matrix including bias term.
        w (np.ndarray):     Weight vector.
        b (float):          Bias term.
        mu (np.ndarray):    Mean value of each feature.
        sigma (np.ndarray): Standard deviation of each feature.
    Returns:
        predicted_value (np.ndarray):  The predicted probabilities.
    """
    X_scaled = (X - mu) / sigma
    predicted_value = model_output(X_scaled, w, b)

    return predicted_value


def predict_scaled(X, w, b, mu, sigma, threshold=0.5):
    """
    Predict the binary class using learned weights and bias.

        Parameters:
        X (np.ndarray):     Feature matrix including bias term.
        w (np.ndarray):     Weight vector.
        b (float):          Bias term.
        mu (np.ndarray):    Mean value of each feature.
        sigma (np.ndarray): Standard deviation of each feature.
        threshold (float):  Classification threshold.

    Returns:
        predicted_value (np.ndarray):  Predicted class labels (0 or 1).
    """
    y_hat = predict_scaled_proba(X, w, b, mu, sigma)

    predicted_value = (y_hat >= threshold).astype(int)

    return predicted_value


def main():
    # Load the training data from the CSV file
    X, y = load_data(TRAINING_DATA_PATH)

    # Split the data into training and testing sets
    X_train, y_train, X_test, y_test = split_data(
        X,
        y,
        test_ratio=0.2,
        seed=SEED,
        shuffle=True
    )

    # Scale features to improve the stability and convergence of gradient descent
    X_train_scaled, mu, sigma = scale_features(X_train)

    # Initialize the model parameters
    w_init = np.zeros(X.shape[1])
    b_init = 0.0

    # Define learning rate and number oif iterations
    alpha = 1.0e-2
    iterations = 10000

    # Train model using gradient descent
    w, b = gradient_descent(X_train_scaled, y_train, w_init, b_init, alpha, iterations)
    print("\nLearned parameters:")
    print("Weights: w =", w)  # w = [-2.74 -0.45 -3.19  0.7  -1.96]
    print(f"Bias: b = {b:.2f}")  # b = 0.18

    # Evaluate the model on the test set (Log Loss / Accuracy)
    y_test_pred = predict_scaled_proba(X_test, w, b, mu, sigma)
    log_loss = compute_cross_entropy_loss(y_test, y_test_pred)
    accuracy = compute_accuracy(y_test, y_test_pred)

    print("\nEvaluation on the test set:")
    print(f"Log Loss: {log_loss:.4f}")  # 0.0876
    print(f"Accuracy: {accuracy:.3f}")  # 1.000

    # Test cars: same specifications, but different price
    test_cars = np.array([
        [80000, 5, 120, 1, 10000],  # cheap (should buy)
        [80000, 5, 120, 1, 50000],  # expensive (should not buy)
        [129358, 12, 211, 1, 16817],  # average (should buy)
    ], dtype=float)

    # Predict, if the user will buy each car
    predictions = predict_scaled(test_cars, w, b, mu, sigma)
    print("\nPredictions for test cars:")
    print(f"Cheap car       -> buy={predictions[0]}")  # 1 = yes
    print(f"Expensive car   -> buy={predictions[1]}")  # 0 = no
    print(f"Average car     -> buy={predictions[2]}")  # 1 = yes


if __name__ == "__main__":
    main()

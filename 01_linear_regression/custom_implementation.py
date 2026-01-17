import numpy as np
from pathlib import Path
import copy

SEED = 42

# Ensure reproducible results
np.random.seed(SEED)

np.set_printoptions(precision=2, suppress=True)

BASE_DIR = Path(__file__).resolve().parent
TRAINING_DATA_PATH = BASE_DIR / "data" / "training_data.csv"


def load_training_data(file_path):
    """
    Loads training data from a CSV file and split it into features and target values.

    Parameters:
        file_path (str | Path): Path to the CSV training data file.

    Returns:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): True target values.
    """
    data = np.loadtxt(fname=file_path, delimiter=',', skiprows=1, dtype=float)
    X = data[:, :-1]
    y = data[:, -1]

    return X, y


def model_output(X, w, b):
    """
    Computes the raw output of the linear regression model f(x) = w·x + b.

    Parameters:
        X (np.ndarray): Feature matrix.
        w (np.ndarray): Weight vector.
        b (float):      Bias term.

    Returns:
        y_hat (np.ndarray): Predicted values for each input sample.
    """

    return X @ w + b


def compute_cost(X, y, w, b):
    """
    Computes the mean squared error (MSE) cost for linear regression.

    Parameters:
        X (np.ndarray):     Feature matrix including.
        y (np.ndarray):     True target values.
        w (np.ndarray):     Weight vector.
        b (float):          Bias term.

    Returns:
        cost (float): The MSE cost value.
    """
    m = len(y)
    y_hat = model_output(X, w, b)
    errors = y_hat - y
    cost = (1 / (2 * m)) * np.sum(errors ** 2)

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
    Performs batch gradient descent to learn weights + bias.

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


def predict_scaled(X, w, b, mu, sigma):
    """
    Predict target values using learned weights and bias.

        Parameters:
        X (np.ndarray):     Feature matrix including bias term.
        w (np.ndarray):     Weight vector.
        b (float):          Bias term.
        mu (np.ndarray):    Mean value of each feature.
        sigma (np.ndarray): Standard deviation of each feature.

    Returns:
        predicted_value (np.ndarray):  The predicted values.
    """
    X_scaled = (X - mu) / sigma
    predicted_value = model_output(X_scaled, w, b)

    return predicted_value


def main():
    # Load the training data from the CSV file
    X, y = load_training_data(TRAINING_DATA_PATH)

    # Scale features to improve the stability and convergence of gradient descent
    X_scaled, mu, sigma = scale_features(X)

    # Initialize the model parameters
    w_init = np.zeros(X.shape[1])
    b_init = 0.0

    # Define learning rate and number oif iterations
    alpha = 1.0e-2
    iterations = 10000

    # Train model using gradient descent
    w, b = gradient_descent(X_scaled, y, w_init, b_init, alpha, iterations)
    print("Learned weights: w =", w)  # w = [-2977.26 -3740.68  3785.22  -992.96]
    print(f"Learned bias: b = {b:.2f}")  # b = 19685.27

    # Predict the price of a test car
    test_car = np.array([80000, 5, 120, 1])
    predicted_price = predict_scaled(test_car, w, b, mu, sigma)

    print(f"Predicted price: {predicted_price:.2f}")  # 20032.71


if __name__ == "__main__":
    main()

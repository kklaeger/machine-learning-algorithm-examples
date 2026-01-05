import numpy as np
from pathlib import Path
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

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


def main():
    # Load data
    X, y = load_training_data(TRAINING_DATA_PATH)
    print(X[0][0])
    # Feature scaling (same idea as in the custom implementation)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Define number oif iterations
    iterations = 1000

    # Linear regression trained via gradient descent
    model = SGDRegressor(max_iter=iterations)

    # Train model using gradient descent
    model.fit(X_scaled, y)

    w = model.coef_
    b = model.intercept_[0]
    print("Learned weights: w =", w)  # w = [-2976.71 -3741.16  3784.52  -993.1]
    print(f"Learned bias: b = {b:.2f}")  # b = 19684.45

    # Predict the price of a test car
    test_car = np.array([[80000, 5, 120, 1]])
    test_car_scaled = scaler.transform(test_car)
    predicted_price = model.predict(test_car_scaled)

    print(f"Predicted price: {predicted_price[0]:.2f}") # 20032.86


if __name__ == "__main__":
    main()

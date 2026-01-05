import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

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
    # Load the training data from the CSV file
    X, y = load_training_data(TRAINING_DATA_PATH)

    # Feature scaling (same idea as in the custom implementation)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Define number oif iterations
    iterations = 10000

    # Logistic Regression
    model = LogisticRegression(max_iter=iterations)

    # Train model
    model.fit(X_scaled, y)

    w = model.coef_
    b = model.intercept_[0]
    print("Learned weights: w =", w)  # w = [[-2.06 -0.39 -2.19  0.47 -1.56]]
    print(f"Learned bias: b = {b:.2f}")  # b = 0.18

    # Two test cars: same specifications, but different price
    test_cars = np.array([
        [80000, 5, 120, 1, 10000],  # cheap
        [80000, 5, 120, 1, 50000],  # expensive
    ], dtype=float)

    test_cars_scaled = scaler.transform(test_cars)

    # Predict, if the user will buy each car
    predictions = model.predict(test_cars_scaled)
    print(f"Cheap car       -> buy={predictions[0]}")  # 1 = yes
    print(f"Expensive car   -> buy={predictions[1]}")  # 0 = no


if __name__ == "__main__":
    main()

import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from utils.data_utils import load_data, split_data
from utils.metrics import compute_cross_entropy_loss, compute_accuracy

SEED = 42

# Ensure reproducible results
np.random.seed(SEED)

np.set_printoptions(precision=2, suppress=True)

BASE_DIR = Path(__file__).resolve().parent
TRAINING_DATA_PATH = BASE_DIR / "data" / "training_data.csv"


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

    # Feature scaling (same idea as in the custom implementation)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Define number of iterations
    iterations = 10000

    # Logistic Regression
    model = LogisticRegression(
        max_iter=iterations,
        random_state=SEED
    )

    # Train model
    model.fit(X_train_scaled, y_train)

    w = model.coef_
    b = model.intercept_[0]
    print("Learned parameters:")
    print("Weights: w =", w)  # w = [[-2.06 -0.39 -2.19  0.47 -1.56]]
    print(f"Bias: b = {b:.2f}")  # b = 0.18

    # Evaluate the model on the test set (Log Loss / Accuracy)
    X_test_scaled = scaler.transform(X_test)
    y_test_pred = model.predict_proba(X_test_scaled)[:, 1]
    log_loss = compute_cross_entropy_loss(y_test, y_test_pred)
    accuracy = compute_accuracy(y_test, y_test_pred)

    print("\nEvaluation on the test set:")
    print(f"Log Loss: {log_loss:.4f}")  # 0.1367
    print(f"Accuracy: {accuracy:.3f}")  # 1.000

    # Test cars: same specifications, but different price
    test_cars = np.array([
        [80000, 5, 120, 1, 10000],  # cheap (should buy)
        [80000, 5, 120, 1, 50000],  # expensive (should not buy)
        [129358, 12, 211, 1, 16817],  # average (should buy)
    ], dtype=float)

    test_cars_scaled = scaler.transform(test_cars)

    # Predict, if the user will buy each car
    predictions = model.predict(test_cars_scaled)
    print("\nPredictions for test cars:")
    print(f"Cheap car       -> buy={int(predictions[0])}")  # 1 = yes
    print(f"Expensive car   -> buy={int(predictions[1])}")  # 0 = no
    print(f"Average car     -> buy={int(predictions[2])}")  # 1 = yes


if __name__ == "__main__":
    main()

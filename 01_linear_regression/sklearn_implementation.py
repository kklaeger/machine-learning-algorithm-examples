import numpy as np
from pathlib import Path
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

from utils.data_utils import load_data, split_data
from utils.metrics import compute_mse_rmse

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

    # Define number oif iterations
    iterations = 1000

    # Linear regression trained via gradient descent
    model = SGDRegressor(
        max_iter=iterations,
        random_state=SEED
    )

    # Train model using gradient descent
    model.fit(X_train_scaled, y_train)

    w = model.coef_
    b = model.intercept_[0]
    print("Learned parameters:")
    print("Weights: w =", w)  # w = [-2976.71 -3741.16  3784.52  -993.1]
    print(f"Bias: b = {b:.2f}")  # b = 19684.45

    # Evaluate the model on the test set (MSE / RMSE)
    X_test_scaled = scaler.transform(X_test)
    y_test_pred = model.predict(X_test_scaled)
    test_mse, test_rmse = compute_mse_rmse(y_test, y_test_pred)

    print("\nEvaluation on the test set:")
    print(f"MSE:  {test_mse:.2f}") # 1436868.78
    print(f"RMSE: {test_rmse:.2f}") # 1198.69

    # Predict the price of a test car
    test_car = np.array([[80000, 5, 120, 1]])
    test_car_scaled = scaler.transform(test_car)
    predicted_price = model.predict(test_car_scaled)

    print("\nPredicting price for test car with features:")
    print(f"Predicted price: {predicted_price[0]:.2f}") # 20032.86


if __name__ == "__main__":
    main()

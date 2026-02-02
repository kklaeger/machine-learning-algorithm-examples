import numpy as np
from pathlib import Path
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from utils.data_utils import load_data
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
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=SEED,
        shuffle=True
    )

    # Feature scaling (same idea as in the custom implementation)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Define number oif iterations
    iterations = 1000

    # Linear regression trained via gradient descent and L2 regularization
    model = SGDRegressor(
        max_iter=iterations,
        random_state=SEED,
        alpha=0.001,
    )

    # Train model using gradient descent
    model.fit(X_train_scaled, y_train)

    w = model.coef_
    b = model.intercept_[0]
    print("Learned parameters:")
    print("Weights: w =", w)  # w = [-3194.22 -3857.84  3652.63 -1099.67]
    print(f"Bias: b = {b:.2f}")  # b = 19895.80

    # Evaluate the model on the test set (MSE / RMSE)
    X_test_scaled = scaler.transform(X_test)
    y_test_pred = model.predict(X_test_scaled)
    test_mse, test_rmse = compute_mse_rmse(y_test, y_test_pred)

    print("\nEvaluation on the test set:")
    print(f"MSE:  {test_mse:.2f}") # 2624124.16
    print(f"RMSE: {test_rmse:.2f}") # 1619.91

    # Predict the price of a test car
    test_car = np.array([[80000, 5, 120, 1]])
    test_car_scaled = scaler.transform(test_car)
    predicted_price = model.predict(test_car_scaled)

    print("\nPredicting price for test car with features:")
    print(f"Predicted price: {predicted_price[0]:.2f}") # 20298.46


if __name__ == "__main__":
    main()

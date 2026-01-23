import numpy as np
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from utils.data_utils import load_data

SEED = 42

# Ensure reproducible results
np.random.seed(SEED)

np.set_printoptions(precision=2, suppress=True)

BASE_DIR = Path(__file__).resolve().parent
TRAINING_DATA_PATH = BASE_DIR / "data" / "training_data.csv"
CROSS_VAL_DATA_PATH = BASE_DIR / "data" / "cross_val_data.csv"


def main():
    # Load the training and cross validation data from the CSV files
    X_train = load_data(TRAINING_DATA_PATH, with_target=False)
    X_val, y_val = load_data(CROSS_VAL_DATA_PATH, with_target=True)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Define the model
    model = IsolationForest(
        n_estimators=100,
        contamination=0.1,
        random_state=SEED
    )

    # Train the model
    model.fit(X_train_scaled)

    y_val_pred = model.predict(X_val_scaled)

    # Convert to 0 = normal, 1 = anomaly (sklearn uses 1 = normal, -1 = anomaly)
    y_val_pred = (y_val_pred == -1).astype(int)

    print("\nEvaluation:")
    print("Predicted anomalies:", int(np.sum(y_val_pred)))
    print("True anomalies:     ", int(np.sum(y_val)))

    # Predict anomalies for test transactions
    test_transactions = np.array([
        [25.0, 14, 2, 1.0],  # normal purchase
        [950.0, 3, 12, 800.0],  # suspicious transaction
        [45.0, 16, 1, 2.5],  # normal purchase
    ], dtype=float)

    # Scale test transactions
    test_transactions_scaled = scaler.transform(test_transactions)

    # Predict anomalies
    test_predictions = model.predict(test_transactions_scaled)
    predictions = (test_predictions == -1).astype(int)

    print("\nPredictions for test transactions:")
    print(f"Transaction 1 -> anomaly={predictions[0]}")  # 0 = normal
    print(f"Transaction 2 -> anomaly={predictions[1]}")  # 1 = anomaly
    print(f"Transaction 3 -> anomaly={predictions[2]}")  # 0 = normal


if __name__ == "__main__":
    main()

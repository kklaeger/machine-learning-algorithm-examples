import numpy as np
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier, export_text

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


def main():
    # Load training data
    X, y = load_training_data(TRAINING_DATA_PATH)

    # Decision Tree Classifier
    model = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=7,
        min_samples_split=10,
        random_state=SEED
    )

    # Train model
    model.fit(X, y)

    # Example patients
    test_patients = np.array([
        [45, 24.5, 95, 0, 4.5],  # low risk
        [52, 31.2, 110, 1, 1.0],  # high risk (obesity + low activity)
        [38, 27.5, 130, 0, 3.0],  # high risk (elevated fasting glucose)
    ], dtype=float)

    predictions = model.predict(test_patients)
    print(f"Patient 1 (low risk)                    -> prediction = {int(predictions[0])}")
    print(f"Patient 2 (obesity + low activity)      -> prediction = {int(predictions[1])}")
    print(f"Patient 3 (elevated fasting glucose)    -> prediction = {int(predictions[2])}")

    feature_names = [
        "age",
        "bmi",
        "fasting_glucose",
        "family_history",
        "physical_activity_hours_per_week"
    ]

    print("\nDecision tree rules:")
    print(export_text(model, feature_names=feature_names))

    print("\nFeature importances:")
    for name, importance in zip(feature_names, model.feature_importances_):
        print(f"  {name}: {importance:.2f}")


if __name__ == "__main__":
    main()

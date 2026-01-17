import numpy as np
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier, export_text

from utils.data_utils import load_data, split_data
from utils.metrics import compute_accuracy

SEED = 42

# Ensure reproducible results
np.random.seed(SEED)

np.set_printoptions(precision=2, suppress=True)

BASE_DIR = Path(__file__).resolve().parent
TRAINING_DATA_PATH = BASE_DIR / "data" / "training_data.csv"


def main():
    # Load training data
    X, y = load_data(TRAINING_DATA_PATH)

    # Split the data into training and testing sets
    X_train, y_train, X_test, y_test = split_data(
        X,
        y,
        test_ratio=0.2,
        seed=SEED,
        shuffle=True
    )

    # Decision Tree Classifier
    model = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=7,
        min_samples_split=10,
        random_state=SEED
    )

    # Train model
    model.fit(X_train, y_train)

    print("\nLearned parameters:")
    print(f"Tree depth: {model.get_depth()}")
    print(f"Number of leaves: {model.get_n_leaves()}")

    # Evaluate on test set
    y_test_prob = model.predict_proba(X_test)[:, 1]
    accuracy = compute_accuracy(y_test, y_test_prob)

    print("\nEvaluation on the test set:")
    print(f"Accuracy: {accuracy:.3f}")

    feature_names = [
        "age",
        "bmi",
        "fasting_glucose",
        "family_history",
        "physical_activity_hours_per_week"
    ]

    # Display decision tree rules and feature importances
    print("\nDecision tree rules:")
    print(export_text(model, feature_names=feature_names))
    print("\nFeature importances:")
    for name, importance in zip(feature_names, model.feature_importances_):
        print(f"  {name}: {importance:.2f}")

    # Example patients
    test_patients = np.array([
        [45, 24.5, 95, 0, 4.5],  # low risk
        [52, 31.2, 110, 1, 1.0],  # high risk (obesity + low activity)
        [38, 27.5, 130, 0, 3.0],  # high risk (elevated fasting glucose)
    ], dtype=float)

    predictions = model.predict(test_patients)
    print("\nPredictions for example patients:")
    print(f"Patient 1 (low risk)                    -> prediction = {int(predictions[0])}")  # 0 low risk
    print(f"Patient 2 (obesity + low activity)      -> prediction = {int(predictions[1])}")  # 1 high risk
    print(f"Patient 3 (elevated fasting glucose)    -> prediction = {int(predictions[2])}")  # 1 high risk


if __name__ == "__main__":
    main()

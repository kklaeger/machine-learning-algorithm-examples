import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from utils.data_utils import load_data, split_data
from utils.metrics import compute_accuracy

SEED = 42

# Ensure reproducible results
np.random.seed(SEED)
tf.random.set_seed(SEED)

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
    normalization_layer = tf.keras.layers.Normalization(axis=-1)
    normalization_layer.adapt(X_train)
    X_train_scaled = normalization_layer(X_train)

    # Define the neural network model
    model = Sequential(
        [
            tf.keras.Input(shape=(5,)),
            Dense(16, activation='relu', name='layer1'),
            # The output layer uses linear activation to improve the stability; sigmoid is applied in the loss function
            Dense(1, activation='linear', name='layer2')
        ]
    )

    # Define learning rate and number oif iterations
    learning_rate = 0.01
    epochs = 100

    # Set up training configuration (loss, optimizer, metrics)
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=["accuracy"]
    )

    # Train model and update weights
    model.fit(X_train_scaled, y_train, epochs=epochs)

    W1, b1 = model.get_layer("layer1").get_weights()
    W2, b2 = model.get_layer("layer2").get_weights()

    print("\nLearned parameters:")
    print("Layer 1 weights:\n", W1)
    print("Layer 1 bias:\n", b1)
    print("Layer 2 weights:\n", W2)
    print("Layer 2 bias:\n", b2)

    # Evaluate the model on the test set
    X_test_scaled = normalization_layer(X_test)
    logits = model.predict(X_test_scaled)
    y_test_prob = tf.sigmoid(logits).numpy().ravel()
    accuracy = compute_accuracy(y_test, y_test_prob)

    print("\nEvaluation on the test set:")
    print(f"Accuracy: {accuracy:.3f}")  # 1.000

    # Test cars: same specifications, but different price
    test_cars = np.array([
        [80000, 5, 120, 1, 10000],  # cheap (should buy)
        [80000, 5, 120, 1, 50000],  # expensive (should not buy)
        [129358, 12, 211, 1, 16817],  # average (should buy)
    ], dtype=float)

    # Scale test cars with the same normalization layer
    test_cars_scaled = normalization_layer(test_cars)

    # Predict, if the user will buy each car
    predictions = model.predict(test_cars_scaled)
    probs = tf.sigmoid(predictions).numpy()
    decisions = (probs >= 0.5).astype(int)

    print("\nPredictions for test cars:")
    print(f"Cheap car       -> buy={decisions[0]}")  # 1 = yes
    print(f"Expensive car   -> buy={decisions[1]}")  # 0 = no
    print(f"Average car     -> buy={decisions[2]}")  # 1 = yes


if __name__ == "__main__":
    main()

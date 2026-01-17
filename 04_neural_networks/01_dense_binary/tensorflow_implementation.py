import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

np.set_printoptions(precision=2, suppress=True)

# Ensure reproducible results
np.random.seed(1234)
tf.random.set_seed(1234)

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
    normalization_layer = tf.keras.layers.Normalization(axis=-1)
    normalization_layer.adapt(X)
    X_scaled = normalization_layer(X)

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
    model.fit(X_scaled, y, epochs=epochs)

    W1, b1 = model.get_layer("layer1").get_weights()
    W2, b2 = model.get_layer("layer2").get_weights()
    print("Layer 1 weights:\n", W1)
    print("Layer 1 bias:\n", b1)
    print("Layer 2 weights:\n", W2)
    print("Layer 2 bias:\n", b2)

    # Test cars: same specifications, but different price
    test_cars = np.array([
        [80000, 5, 120, 1, 10000],  # cheap (should buy)
        [80000, 5, 120, 1, 50000],  # expensive (should not buy)
        [129358, 12, 211, 1, 16817],  # average (should buy)
    ], dtype=float)

    test_cars_scaled = normalization_layer(test_cars)

    # Predict, if the user will buy each car
    predictions = model.predict(test_cars_scaled)
    decisions = (predictions >= 0.5).astype(int)
    print(f"Cheap car       -> buy={decisions[0]}")  # 1 = yes
    print(f"Expensive car   -> buy={decisions[1]}")  # 0 = no
    print(f"Average car     -> buy={decisions[2]}")  # 1 = yes


if __name__ == "__main__":
    main()

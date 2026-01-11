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

    # Define model input and output dimensions
    number_of_features = X_scaled.shape[1]  # should be 4
    number_of_classes = 3

    # Define the neural network model
    model = Sequential(
        [
            tf.keras.Input(shape=(number_of_features,)),
            Dense(16, activation='relu', name='layer1'),
            # Output layer uses linear activation; softmax is applied outside (and loss uses from_logits=True)
            Dense(number_of_classes, activation='linear', name='layer2')
        ]
    )

    # Define learning rate and number oif iterations
    learning_rate = 0.01
    epochs = 100

    # Set up training configuration (loss, optimizer, metrics)
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
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

    # Test cars for multiclass price category prediction
    test_cars = np.array([
        [210000, 14, 95, 3],  # very high mileage, old, weak engine -> cheap
        [125000, 8, 180, 1],  # medium mileage, decent power -> average
        [35000, 2, 360, 0],  # low mileage, young, powerful -> expensive
    ], dtype=float)

    # Scale test cars with the same normalization layer
    test_cars_scaled = normalization_layer(test_cars)

    # Predict, to which class each car belongs
    probabilities = model.predict(test_cars_scaled)
    predicted_classes = np.argmax(probabilities, axis=1)
    labels = np.array(["cheap", "average", "expensive"])
    print("Car 1 ->", labels[predicted_classes[0]])
    print("Car 2 ->", labels[predicted_classes[1]])
    print("Car 3 ->", labels[predicted_classes[2]])


if __name__ == "__main__":
    main()

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

    # Define model input and output dimensions
    number_of_features = X_train_scaled.shape[1]  # should be 4
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
    y_test_prob = tf.nn.softmax(logits, axis=1).numpy()
    accuracy = compute_accuracy(y_test, y_test_prob)

    print("\nEvaluation on the test set:")
    print(f"Accuracy: {accuracy:.3f}")  # e.g. 0.850

    # Test cars for multiclass price category prediction
    test_cars = np.array([
        [210000, 14, 95, 3],  # very high mileage, old, weak engine -> cheap
        [125000, 8, 180, 1],  # medium mileage, decent power -> average
        [35000, 2, 360, 0],  # low mileage, young, powerful -> expensive
    ], dtype=float)

    # Scale test cars with the same normalization layer
    test_cars_scaled = normalization_layer(test_cars)

    # Predict, to which class each car belongs
    logits = model.predict(test_cars_scaled)
    predicted_classes = np.argmax(logits, axis=1)
    labels = np.array(["cheap", "average", "expensive"])

    print("\nPredictions for test cars:")
    print("Car 1 ->", labels[predicted_classes[0]])  # cheap
    print("Car 2 ->", labels[predicted_classes[1]])  # average
    print("Car 3 ->", labels[predicted_classes[2]])  # expensive


if __name__ == "__main__":
    main()

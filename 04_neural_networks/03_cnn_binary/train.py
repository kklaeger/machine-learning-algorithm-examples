import os
import numpy as np
import tensorflow as tf

from data import load_raw_data, prepare_dataset
from model import build_model

# Ensure reproducible results
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

MAX_EPOCHS = 50
MODEL_PATH = "models/model.keras"


def train_model(model, training_data, validation_data, epochs, best_model_path):
    """
    Trains the CNN model on the training dataset and validates it on the validation dataset.

    Parameters:
        - model (tf.keras.Model): The CNN model to be trained.
        - training_data (tf.data.Dataset): The training dataset.
        - validation_data (tf.data.Dataset): The validation dataset.
        - epochs (int): Number of epochs to train the model.
    Returns:
        - model (tf.keras.Model):               The trained CNN model.
        - history (tf.keras.callbacks.History): Training history object containing loss and accuracy metrics.
    """
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

    # Using callbacks for early stopping and model checkpointing
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(filepath=best_model_path, monitor="val_loss", save_best_only=True),
    ]

    # Train the model
    history = model.fit(
        training_data,
        validation_data=validation_data,
        epochs=epochs,
        callbacks=callbacks
    )
    return model, history


def main():
    # Load the raw dataset
    training_data_raw, validation_data_raw, _, data_info = load_raw_data()

    # Prepare datasets
    training_data = prepare_dataset(training_data_raw, batch_size=32, shuffle=True, do_batch=True)
    validation_data = prepare_dataset(validation_data_raw, batch_size=32, shuffle=False, do_batch=True)

    print(f"Dataset: {data_info.name}")
    print(f"Number of training samples: {training_data_raw.cardinality().numpy()}")
    print(f"Number of validation samples: {validation_data_raw.cardinality().numpy()}")

    # Build the model
    model = build_model(
        input_shape=(160, 160, 3)  # Height, Width, RGB channels
    )
    model.summary()

    # Train the model
    train_model(
        model=model,
        training_data=training_data,
        validation_data=validation_data,
        epochs=MAX_EPOCHS,  # Using max epochs because of early stopping callback
        best_model_path=MODEL_PATH
    )


if __name__ == "__main__":
    main()

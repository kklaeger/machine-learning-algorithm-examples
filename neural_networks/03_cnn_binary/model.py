import tensorflow as tf

LEARNING_RATE = 2e-4


def build_model(input_shape=(160, 160, 3)):
    """
    Builds a simple Convolutional Neural Network (CNN) model for binary classification.

    Parameters:
        - input_shape (tuple): 3D shape of the input images (height, width, RGB channels).

    Returns:
        - model (tf.keras.Model): Compiled CNN model.
    """
    data_augmentation = tf.keras.models.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.1),
    ], name="data_augmentation")

    model = tf.keras.models.Sequential([
        # Explicit input layer defining the expected image shape
        tf.keras.layers.Input(shape=input_shape),

        # Data augmentation layers to improve model generalization layers.
        data_augmentation,

        # First convolutional block using 32 filters with ReLU activation.
        # Learns low-level visual features (e.g. edges, corners, color contrasts).
        # Max pooling reduces spatial dimensions while retaining the most salient features.
        tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),

        # Second convolutional block using 64 filters with ReLU activation.
        # Learns more complex patterns by combining low-level features.
        # Max pooling further reduces spatial resolution and increases feature abstraction.
        tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),

        # Third convolutional block using 128 filters with ReLU activation.
        # Captures high-level visual features and object-specific patterns.
        # Max pooling provides additional spatial downsampling and robustness.
        tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),

        # Global average pooling layer to convert 3D feature maps to 1D feature vectors.
        # Reduces the number of parameters and helps prevent overfitting.
        tf.keras.layers.GlobalAveragePooling2D(),

        # Fully connected layer to learn complex combinations of the extracted features.
        tf.keras.layers.Dense(units=128, activation="relu"),

        # Dropout layer to reduce overfitting by randomly setting a fraction of input units to 0 during training.
        tf.keras.layers.Dropout(0.3),

        # Output layer with a single neuron and sigmoid activation for binary classification.
        tf.keras.layers.Dense(units=1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy")]
    )

    return model

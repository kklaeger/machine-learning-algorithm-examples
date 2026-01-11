import numpy as np
from pathlib import Path

np.set_printoptions(precision=2, suppress=True)

# Ensure reproducible results
np.random.seed(1234)

BASE_DIR = Path(__file__).resolve().parent
TRAINING_DATA_PATH = BASE_DIR / "data" / "training_data.csv"


# -------- Helpers Functions --------
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


def one_hot(y, num_classes):
    """
    Convert class labels to one-hot encoded vectors.

    Parameters:
        y (np.ndarray): Array of class labels of shape (m,).
        num_classes (int): Number of classes.
    Returns:
        y_one_hot (np.ndarray): One-hot encoded matrix of shape (m, num_classes).
    """
    y = y.astype(int).reshape(-1)
    out = np.zeros((y.size, num_classes))
    out[np.arange(y.size), y] = 1
    return out


# -------- Activation Functions --------
def softmax(z):
    """
    Apply the softmax activation function.

    Parameters:
        z (np.ndarray): Input array of shape (m, n_classes).

    Returns:
        softmax (np.ndarray): Softmax-transformed output of shape (m, n_classes).
    """
    z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def relu(z):
    """
    Apply the ReLU activation function.

    Parameters:
        z (np.ndarray | float): Input value(s).

    Returns:
        relu (np.ndarray | float): ReLU-transformed output.
    """
    return np.maximum(0, z)


def relu_backward(dA, z):
    """
    Backward pass for the ReLU activation function.

    Parameters:
        dA (np.ndarray):    Gradient of the loss with respect to the activation output.
        z (np.ndarray):     Pre-activation values.
    Returns:
        dZ (np.ndarray): Gradient of the loss with respect to the pre-activation values.
    """
    dZ = np.array(dA, copy=True)
    return dZ * (z > 0)


# -------- Loss Functions --------
def categorical_cross_entropy(y_hat, y):
    """
    Compute the categorical cross-entropy loss.

    Parameters:
        y_hat (np.ndarray): Predicted probabilities of shape (m, n_classes).
        y (np.ndarray):     True target values (one-hot encoded) of shape (m, n_classes).

    Returns:
        loss (float): Categorical cross-entropy loss.
    """
    # Avoid log(0) by clipping probabilities
    eps = 1e-15
    y_hat = np.clip(y_hat, eps, 1 - eps)

    return -np.mean(np.sum(y * np.log(y_hat), axis=1))


# -------- Layers --------
class NormalizationLayer:
    """
    A layer that normalizes input data to have zero mean and unit variance.
    """

    def __init__(self):
        self.mu = None
        self.sigma = None
        self.adapted = False

    def adapt(self, X):
        """
        Learn per-feature mean and standard deviation from the training data.

        Parameters:
            X (np.ndarray): Feature matrix.
        """
        self.mu = np.mean(X, axis=0)
        self.sigma = np.std(X, axis=0)
        self.adapted = True

    def __call__(self, X):
        """
        Apply feature-wise normalization using previously learned statistics.

        Parameters:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).

        Returns:
            X_normalized (np.ndarray): Normalized feature matrix.
        """
        if not self.adapted:
            raise RuntimeError("NormalizationLayer must be adapted before calling.")

        return (X - self.mu) / self.sigma


class DenseLayer:
    """
    A fully connected (dense) layer.
    """

    def __init__(self, input_dim, output_dim, activation, name):

        # Initialize weights and biases
        self.W = 0.01 * np.random.randn(input_dim, output_dim)
        self.b = np.zeros((1, output_dim))

        self.activation = activation
        self.name = name

        # Caches for the backward propagation
        self.A_prev = None
        self.Z = None

        # gradients (computed in backward propagation)
        self.dW = None
        self.db = None

    def forward(self, A_prev):
        """
        Forward propagation through the dense layer using the specified activation function.

        Computes Z = A_prev * W + b and then applies the activation function.

        Stores A_prev and Z for use in backward propagation.

        Parameters:
            A_prev (np.ndarray): Activations from the previous layer of shape (m, input_dim).

        Returns:
            A (np.ndarray): Activations from this layer of shape (m, output_dim).
        """
        # Cache A_prev for the backward propagation
        self.A_prev = A_prev

        # Linear step
        self.Z = A_prev @ self.W + self.b

        # Activation step
        if self.activation == "relu":
            return relu(self.Z)
        if self.activation == "linear":
            return self.Z
        raise ValueError(f"Unknown activation: {self.activation}")

    def backward(self, dA):
        """
        Backward propagation through the dense layer.

        Computes gradients of the loss with respect to:
        - the layer's weights (dW)
        - the layer's biases (db)
        - the input activations (dA_prev)

        Parameters:
            dA (np.ndarray): Gradient of the loss with respect to this layer's output (after activation).

        Returns:
            dA_prev (np.ndarray): Gradient of the loss with respect to the input activations of this layer.
        """

        # Convert gradient after activation (dA) to gradient before activation (dZ)
        if self.activation == "relu":
            dZ = relu_backward(dA, self.Z)
        elif self.activation == "linear":
            dZ = dA
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

        # Compute the gradients for weights and bias
        m = self.A_prev.shape[0]
        self.dW = (1 / m) * (self.A_prev.T @ dZ)
        self.db = (1 / m) * np.sum(dZ, axis=0, keepdims=True)

        # Compute dA_prev to propagate the gradient to the previous layer
        dA_prev = dZ @ self.W.T
        return dA_prev

    def update(self, learning_rate):
        """
        Update weights and biases using the computed gradients and the specified learning rate. This is the actual gradient descent step.

        Parameters:
            learning_rate (float): Learning rate for the update step.
        """
        self.W -= learning_rate * self.dW
        self.b -= learning_rate * self.db

    def get_weights(self):
        """
        Get the current weights and biases of the layer.

        Returns:
            W (np.ndarray): Weights matrix.
            b (np.ndarray): Bias vector.
        """
        return self.W, self.b


# -------- Model --------
class Model:
    """
    A simple sequential model for multiclass classification using dense layers.
    """

    def __init__(self, number_of_classes):
        self.layers = []
        self.number_of_classes = number_of_classes
        self.learning_rate = None

    def add(self, layer):
        """
        Add a layer to the model.

        Parameters:
            layer (DenseLayer): The layer to add to the model.
        """
        self.layers.append(layer)

    def compile(self, learning_rate=1e-2):
        """
        Compile the model by setting the learning rate.

        Parameters:
            learning_rate (float): Learning rate for training.
        """
        self.learning_rate = learning_rate

    def forward(self, X):
        """
        Forward propagation through all layers of the model.

        Parameters:
            X (np.ndarray): Input feature matrix of shape (m, n_features).
        Returns:
            A (np.ndarray): Output of the last layer (logits) of shape (m, 1).
        """
        A = X
        for layer in self.layers:
            A = layer.forward(A)
        return A

    def fit(self, X, y, epochs=100, print_every=10):
        """
        Train the model using batch gradient descent.

        Parameters:
            X (np.ndarray):     Input feature matrix (m, n_features).
            y (np.ndarray):     True target values (m,).
            epochs (int):       Number of training epochs.
            print_every (int):  Frequency of printing the loss during training.
        """
        if self.learning_rate is None:
            raise RuntimeError("Call compile() before fit().")

        # Ensure y is a column vector of shape (m, 1)
        y = y.reshape(-1, 1)

        for epoch in range(1, epochs + 1):
            # 1. Forward propagation through all layers
            y_one_hot = one_hot(y, self.number_of_classes)

            logits = self.forward(X)
            y_hat = softmax(logits)

            # 2. Compute loss
            loss = categorical_cross_entropy(y_hat, y_one_hot)

            # 3. Backward propagation
            # For sigmoid + BCE, the gradient w.r.t. logits simplifies to: dZ_last = y_hat - y_one_hot
            dZ_last = (y_hat - y_one_hot)

            # 4. Backward propagation through all layers
            dA = dZ_last
            for layer in reversed(self.layers):
                dA = layer.backward(dA)

            # 5. Update weights and biases for all layers
            for layer in self.layers:
                layer.update(self.learning_rate)

            if epoch % print_every == 0 or epoch == 1:
                print(f"Epoch {epoch:4d} | Loss: {loss:.6f}")

    def predict(self, X):
        """
        Make predictions using the trained model.

        Parameters:
            X (np.ndarray): Input feature matrix of shape (m, n_features).

        Returns:
            predictions (np.ndarray): Predicted softmax probabilities of shape (m, n_classes).
        """
        logits = self.forward(X)
        predictions = softmax(logits)
        return predictions

    def get_layer(self, name):
        """
        Retrieve a layer by its name.

        Parameters:
            name (str): Name of the layer to retrieve.

        Returns:
            layer (DenseLayer): The layer with the specified name.
        """
        for layer in self.layers:
            if layer.name == name:
                return layer
        raise ValueError(f"No layer named '{name}'")


def main():
    # Load the training data from the CSV file
    X, y = load_training_data(TRAINING_DATA_PATH)

    # Scale features to improve the stability and convergence of gradient descent
    normalization_layer = NormalizationLayer()
    normalization_layer.adapt(X)
    X_scaled = normalization_layer(X)

    # Define model input and output dimensions
    number_of_features = X_scaled.shape[1]  # should be 4
    number_of_classes = 3

    # Define the neural network model
    model = Model(number_of_classes)
    model.add(DenseLayer(input_dim=number_of_features, output_dim=16, activation="relu", name="layer1"))
    # The output layer uses linear activation to improve the stability; sigmoid is applied in the loss function
    model.add(DenseLayer(input_dim=16, output_dim=3, activation="linear", name="layer2"))

    # Define learning rate and number of iterations
    learning_rate = 0.05
    epochs = 1000

    model.compile(learning_rate=learning_rate)
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

import numpy as np
from pathlib import Path

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


def entropy(y):
    """
    Calculate the entropy of a binary label vector y (values 0/1).

    Parameters:
        y (np.ndarray): Binary class labels (0/1).
    Returns:
        entropy (float): Entropy value [0, 1].
    """
    if len(y) == 0:
        return 0.0

    p1 = np.sum(y) / len(y)

    if p1 == 0 or p1 == 1:
        return 0.0

    return -p1 * np.log2(p1) - (1 - p1) * np.log2(1 - p1)


def information_gain(y, y_left, y_right):
    """
    Calculate the information gain of a split.

    Parameters:
        y (np.ndarray):         Original binary class labels (0/1).
        y_left (np.ndarray):    Binary class labels of the left split.
        y_right (np.ndarray):   Binary class labels of the right split.
    Returns:
        info_gain (float): Information gain value.
    """
    if len(y) == 0 or len(y_left) == 0 or len(y_right) == 0:
        return 0.0

    H_parent = entropy(y)
    H_left = entropy(y_left)
    H_right = entropy(y_right)

    w_left = len(y_left) / len(y)
    w_right = len(y_right) / len(y)

    return H_parent - (w_left * H_left + w_right * H_right)


def best_split(X, y):
    """
    Find the feature and threshold that yield the highest information gain for splitting the dataset at a decision
    tree node.

    Parameters:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        y (np.ndarray): Binary class labels of shape (n_samples,).

    Returns:
        best_feature (int | None):      Index of the best feature to split on, or None if no valid split exists.
        best_threshold (float | None):  Threshold value for the best split, or None if no valid split exists.
        best_info_gain (float):         Highest information gain achieved by the best split.
    """

    n_features = X.shape[1]
    best_info_gain = -1
    best_feature = None
    best_threshold = None
    for feature_index in range(n_features):
        thresholds = np.unique(X[:, feature_index])
        for threshold in thresholds:
            left_indices = X[:, feature_index] <= threshold
            right_indices = X[:, feature_index] > threshold

            # Skip useless splits that result in empty nodes
            if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
                continue

            y_left = y[left_indices]
            y_right = y[right_indices]

            info_gain = information_gain(y, y_left, y_right)

            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_feature = feature_index
                best_threshold = threshold

    return best_feature, best_threshold, best_info_gain


class Node:
    """
    A node in the decision tree.

    Attributes:
        feature (int):      Index of the feature to split on.
        threshold (float):  Threshold value for the split.
        left (Node):        Left child node.
        right (Node):       Right child node.
        value (int):        Class label for leaf nodes.
    """

    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        """
        Check if the node is a leaf node.

        Returns:
            is_leaf_node (bool): True if the node is a leaf node, False otherwise.
        """
        return self.value is not None


def build_tree(X, y, depth=0, max_depth=5, min_samples_split=2):
    """
    Recursively build a binary decision tree using information gain.

    The recursion stops when:
    - The node is pure (all samples belong to the same class).
    - The maximum depth (max_depth) is reached.
    - The number of samples is less than the minimum required to split (min_samples_split).
    - No valid split is found that improves information gain.

    Parameters:
        X (np.ndarray):             Feature matrix.
        y (np.ndarray):             True target values.
        depth (int):                Current depth of the tree.
        max_depth (int):            Maximum depth of the tree.
        min_samples_split (int):    Minimum number of samples required to split a node.

    Returns:
        node (Node): The root node of the decision tree.
    """
    majority_class = int(np.mean(y) >= 0.5)

    # Stop for an empty node
    if len(y) == 0:
        return Node(value=0)

    # Stop for a pure node
    if np.all(y == y[0]):
        return Node(value=int(y[0]))

    # Stop, if max depth is reached or not enough samples to split
    if depth >= max_depth or len(y) < min_samples_split:
        return Node(value=majority_class)

    # Find the best split
    best_feature, best_threshold, best_info_gain = best_split(X, y)

    # Stop, if there is no valid split or no improvement in information gain
    if best_feature is None or best_threshold is None or best_info_gain <= 0:
        return Node(value=majority_class)

    # Split the dataset
    left_indices = X[:, best_feature] <= best_threshold
    right_indices = X[:, best_feature] > best_threshold

    X_left, y_left = X[left_indices], y[left_indices]
    X_right, y_right = X[right_indices], y[right_indices]

    # Recursively build the left and right subtrees
    left_subtree = build_tree(X_left, y_left, depth + 1, max_depth, min_samples_split)
    right_subtree = build_tree(X_right, y_right, depth + 1, max_depth, min_samples_split)

    # Return the current node
    return Node(feature=best_feature, threshold=best_threshold, left=left_subtree, right=right_subtree)


def predict_one(node, x):
    """
    Predict the class label for a single sample using the decision tree.

    Parameters:
        node (Node):    The root node of the decision tree.
        x (np.ndarray): A single sample feature vector.

    Returns:
        prediction (int): Predicted class label (0 or 1).
    """
    if node.is_leaf_node():
        return node.value

    if x[node.feature] <= node.threshold:
        return predict_one(node.left, x)
    else:
        return predict_one(node.right, x)


def predict(node, X):
    """
    Predict the class labels for multiple samples using the decision tree.

    Parameters:
        node (Node):     The root node of the decision tree.
        X (np.ndarray):  Feature matrix of shape (n_samples, n_features).

    Returns:
        predictions (np.ndarray): Predicted class labels (0 or 1) for each sample.
    """
    predictions = [predict_one(node, x) for x in X]
    return np.array(predictions)


class CustomDecisionTreeClassifier:
    """
    A custom implementation of a Decision Tree Classifier.
    """

    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        """
        Fit the decision tree classifier to the training data.

        Parameters:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): True target values.
        """
        self.tree = build_tree(X, y, max_depth=self.max_depth, min_samples_split=self.min_samples_split)

    def predict(self, X):
        """
        Predict the class labels for multiple samples.

        Parameters:
            X (np.ndarray): Feature matrix.

        Returns:
            predictions (np.ndarray): Predicted class labels (0 or 1) for each sample.
        """
        return predict(self.tree, X)


def print_tree(node, feature_names, depth=0):
    """
    Print the decision tree rules in a human-readable format.

    Parameters:
        node (Node):               The root node of the decision tree.
        feature_names (list):      List of feature names.
        depth (int):               Current depth of the tree (used for indentation).
    """
    indent = "  " * depth

    if node.is_leaf_node():
        print(f"{indent}Predict: {node.value}")
        return

    feature_name = feature_names[node.feature]
    print(f"{indent}if {feature_name} <= {node.threshold:.2f}:")
    print_tree(node.left, feature_names, depth + 1)
    print(f"{indent}else:  # {feature_name} > {node.threshold:.2f}")
    print_tree(node.right, feature_names, depth + 1)


def main():
    # Load training data
    X, y = load_training_data(TRAINING_DATA_PATH)

    # Custom Decision Tree Classifier
    model = CustomDecisionTreeClassifier(
        max_depth=7,
        min_samples_split=10,
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
    print_tree(model.tree, feature_names)


if __name__ == "__main__":
    main()

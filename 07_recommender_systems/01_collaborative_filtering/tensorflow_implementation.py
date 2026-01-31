import numpy as np
import tensorflow as tf

from tensorflow import keras
from pathlib import Path

SEED = 42

# Ensure reproducible results
np.random.seed(SEED)
tf.random.set_seed(42)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "ratings.csv"


def load_data(file_path):
    """
    Loads a rating matrix Y and creates the corresponding indicator matrix R.

    Parameters:
        file_path (str | Path): Path to the CSV file.

    Returns:
        Y (np.ndarray): Rating matrix with missing values set to 0.
        R (np.ndarray): Indicator matrix (1 if rating exists, 0 otherwise).
    """
    data = np.genfromtxt(
        fname=file_path,
        delimiter=",",
        skip_header=1,
        dtype=float
    )
    Y = data[:, 1:]
    R = (Y != 0).astype(np.float32)
    Y = Y.astype(np.float32)

    return Y, R


def compute_cost(X, W, b, Y, R, lambda_):
    """
    Computes the regularized cost function for collaborative filtering.

    Parameters:
        X (tf.Variable): Song features (num_songs, num_features).
        W (tf.Variable): User features (num_users, num_features).
        b (tf.Variable): User bias (num_users,).
        Y (tf.Tensor): Rating matrix (num_songs, num_users).
        R (tf.Tensor): Indicator matrix (num_songs, num_users).
        lambda_ (float): Regularization parameter.
    Returns:
        cost (float): The regularized cost value.
    """
    pred = tf.linalg.matmul(X, W, transpose_b=True) + b
    error = (pred - Y) * R  # consider only rated songs
    cost = 0.5 * tf.reduce_sum(tf.square(error))
    regularization = (lambda_ / 2) * (
            tf.reduce_sum(tf.square(X)) +
            tf.reduce_sum(tf.square(W))
    )

    return cost + regularization


def normalize_ratings(Y, R):
    """
    Normalizes the ratings by subtracting the mean rating for each song.

    Parameters:
        Y (np.ndarray): Rating matrix (num_songs, num_users).
        R (np.ndarray): Indicator matrix (num_songs, num_users).
    Returns:
        Y_norm (np.ndarray): Normalized rating matrix.
        Y_mean (np.ndarray): Mean ratings for each song.
    """
    Y = Y.astype(np.float32)
    R = R.astype(np.float32)
    Y_mean = (np.sum(Y * R, axis=1) / (np.sum(R, axis=1) + 1e-12)).reshape(-1, 1)
    Y_norm = Y - Y_mean * R

    return Y_norm.astype(np.float32), Y_mean.astype(np.float32)


def train_collaborative_filtering(Y, R, num_features=10, learning_rate=1e-1, lambda_=1.0, iterations=100):
    """
    Trains a collaborative filtering model using TensorFlow.

    Parameters:
        Y (np.ndarray): Rating matrix (num_songs, num_users).
        R (np.ndarray): Indicator matrix (num_songs, num_users).
        num_features (int): Number of latent features.
        learning_rate (float): Learning rate for the optimizer.
        lambda_ (float): Regularization parameter.
        iterations (int): Number of training iterations.

    Returns:
        X (tf.Variable): Learned song features (num_songs, num_features).
        W (tf.Variable): Learned user features (num_users, num_features).
        b (tf.Variable): Learned user bias (num_users,).
        cost_history (list): History of cost values during training.
    """

    num_songs, num_users = Y.shape

    Y = tf.convert_to_tensor(Y, dtype=tf.float32)
    R = tf.convert_to_tensor(R, dtype=tf.float32)

    X = tf.Variable(tf.random.normal((num_songs, num_features), dtype=tf.float32))
    W = tf.Variable(tf.random.normal((num_users, num_features), dtype=tf.float32))
    b = tf.Variable(tf.zeros((num_users,), dtype=tf.float32))

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    cost_history = []

    for iteration in range(iterations):
        with tf.GradientTape() as tape:
            cost = compute_cost(X, W, b, Y, R, lambda_)

        grads = tape.gradient(cost, [X, W, b])
        optimizer.apply_gradients(zip(grads, [X, W, b]))

        cost_history.append(cost.numpy())

        if iteration % 10 == 0:
            print(f"Iteration {iteration}: cost = {cost.numpy():.4f}")

    return X, W, b, cost_history


def predict_for_user(X, W, b, Y_mean, R, user_index, top_n=5):
    """
    Predict ratings and recommend songs for a specific user.

    Parameters:
        X (tf.Variable):        Song features (num_songs, num_features)
        W (tf.Variable):        User features (num_users, num_features)
        b (tf.Variable):        User bias (num_users,)
        Y_mean (np.ndarray):    Mean rating per song (num_songs, 1)
        R (np.ndarray):         Indicator matrix (num_songs, num_users)
        user_index (int):       Index of the user
        top_n (int):            Number of recommendations

    Returns:
        recommended_indices (np.ndarray):   Indices of recommended songs
        recommended_scores (np.ndarray):    Predicted ratings
    """
    # Compute the full prediction matrix
    pred_norm = tf.linalg.matmul(X, W, transpose_b=True) + b
    pred = pred_norm + Y_mean

    # Get the predictions for the specific user
    user_pred = pred[:, user_index]

    # Filter out already rated songs
    not_rated = tf.equal(R[:, user_index], 0)
    candidate_preds = tf.boolean_mask(user_pred, not_rated)

    # Get the indices of songs not rated
    song_indices = tf.where(not_rated)[:, 0]

    # If there are fewer candidates than top_n, adjust top_n accordingly
    k = tf.minimum(top_n, tf.shape(candidate_preds)[0])

    # If no candidates, return empty arrays
    if k == 0:
        return tf.constant([], dtype=tf.int32), tf.constant([], dtype=tf.float32)

    # Get top N recommendations
    scores, idx = tf.math.top_k(candidate_preds, k=k)
    recommended_indices = tf.gather(song_indices, idx)

    return recommended_indices, scores


def compute_accuracy(y_true, y_pred, R_mask):
    """
    Computes the accuracy for binary and multi-class classification. Accepts both class labels and predicted
    probabilities.

    Parameters:
        y_true (np.ndarray): True target values.
        y_pred (np.ndarray): Predicted target values.
        R_mask (np.ndarray): Indicator matrix (1 if rating exists, 0 otherwise).
    Returns:
        accuracy (float):    Accuracy score.
    """

    diff = (y_pred - y_true) * R_mask
    return np.sqrt(np.sum(diff ** 2) / np.sum(R_mask))


def main():
    # Load the training data from the CSV file
    Y, R = load_data(DATA_PATH)

    # Get the original number of users and songs
    num_songs, num_users = Y.shape

    # Define custom ratings for a new user to demonstrate recommendations
    my_ratings = np.zeros(num_songs)
    my_ratings[0] = 5
    my_ratings[3] = 4
    my_ratings[6] = 3
    my_ratings[11] = 4
    my_ratings[15] = 5
    my_ratings[18] = 4

    # Add the new user's ratings to the end of the dataset
    Y = np.c_[Y, my_ratings]
    R = np.c_[R, (my_ratings != 0).astype(np.float32)]

    # Update the number of users after adding the new user
    num_users = Y.shape[1]

    # Normalize ratings
    Y_norm, Y_mean = normalize_ratings(Y, R)

    # Train collaborative filtering model
    X, W, b, cost_history = train_collaborative_filtering(
        Y_norm, R,
        num_features=10,
        iterations=1000
    )

    # Evaluate training accuracy
    Y_pred_norm = tf.linalg.matmul(X, W, transpose_b=True) + b
    Y_pred = Y_pred_norm + Y_mean
    train_accuracy = compute_accuracy(Y, Y_pred.numpy(), R)
    print(f"\nTraining RMSE: {train_accuracy:.4f}")

    # Predict and recommend songs for the new user
    last_user_index = num_users - 1
    rec_songs, rec_scores = predict_for_user(
        X, W, b, Y_mean, R,
        user_index=last_user_index,
        top_n=5
    )

    print("\nRecommended song indices:", rec_songs.numpy())
    print("Predicted scores for recommended songs:", rec_scores.numpy())


if __name__ == "__main__":
    main()

import tensorflow as tf
import numpy as np
import pandas as pd

from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pathlib import Path

SEED = 42

# Ensure reproducible results
np.random.seed(SEED)
tf.random.set_seed(42)

BASE_DIR = Path(__file__).resolve().parent
USERS_DATA_PATH = BASE_DIR / "data" / "users.csv"
SONGS_DATA_PATH = BASE_DIR / "data" / "songs.csv"
RATINGS_DATA_PATH = BASE_DIR / "data" / "ratings.csv"

USER_FEATURE_COLUMNS = ["rock", "classical", "pop", "jazz", "hiphop"]
SONG_FEATURE_COLUMNS = ["year", "avg_rating", "rock", "classical", "pop", "jazz", "hiphop"]


def load_raw_data(user_path, song_path, ratings_path):
    """
    Load raw data from CSV files.

    Parameters:
        user_path (str | Path):     Path to the users CSV file.
        song_path (str | Path):     Path to the songs CSV file.
        ratings_path (str | Path):  Path to the ratings CSV file.

    Returns:
        users_df (pd.DataFrame):    DataFrame containing user features.
        songs_df (pd.DataFrame):    DataFrame containing song features.
        ratings_df (pd.DataFrame):  DataFrame containing user-song ratings.
    """
    users_df = pd.read_csv(user_path)
    songs_df = pd.read_csv(song_path)
    ratings_df = pd.read_csv(ratings_path)
    return users_df, songs_df, ratings_df


def process_data(users_df, songs_df, ratings_df, user_features, song_features):
    """
    Process raw data into feature matrices.

    Parameters:
        users_df (pd.DataFrame):    DataFrame containing user features.
        songs_df (pd.DataFrame):    DataFrame containing song features.
        ratings_df (pd.DataFrame):  DataFrame containing user-song ratings.
        user_features (list):       List of user feature column names.
        song_features (list):       List of song feature column names.
    Returns:
        V_u (np.ndarray): User feature matrix.
        V_s (np.ndarray): Song feature matrix.
        y (np.ndarray):   Ratings vector.
    """
    user_vectors = {
        row.user_id: row[user_features].values.astype(float)
        for _, row in users_df.iterrows()
    }

    song_vectors = {
        row.song_id: row[song_features].values.astype(float)
        for _, row in songs_df.iterrows()
    }
    V_u = []
    V_s = []
    y = []

    for _, row in ratings_df.iterrows():
        V_u.append(user_vectors[row.user_id])
        V_s.append(song_vectors[row.song_id])
        y.append(row.rating)

    V_u = np.array(V_u)
    V_s = np.array(V_s)
    y = np.array(y)

    return V_u, V_s, y


def scale_data(scalers, V_u_train, V_u_test, V_s_train, V_s_test, y_train, y_test):
    """
    Scale user features, song features, and ratings using provided scalers.

    Parameters:
        scalers (dict):         Dictionary containing 'user_scaler', 'song_scaler', and 'rating_scaler'.
        V_u_train (np.ndarray): Training user feature matrix.
        V_u_test (np.ndarray):  Testing user feature matrix.
        V_s_train (np.ndarray): Training song feature matrix.
        V_s_test (np.ndarray):  Testing song feature matrix.
        y_train (np.ndarray):   Training ratings vector.
        y_test (np.ndarray):    Testing ratings vector.
    Returns:
        V_u_train (np.ndarray): Scaled training user feature matrix.
        V_u_test (np.ndarray):  Scaled testing user feature matrix.
        V_s_train (np.ndarray): Scaled training song feature matrix.
        V_s_test (np.ndarray):  Scaled testing song feature matrix.
        y_train (np.ndarray):   Scaled training ratings vector.
        y_test (np.ndarray):    Scaled testing ratings vector.
    """
    # Unpack scalers
    user_scaler = scalers["user_scaler"]
    song_scaler = scalers["song_scaler"]
    rating_scaler = scalers["rating_scaler"]

    # Fit scalers on training data only
    user_scaler.fit(V_u_train)
    song_scaler.fit(V_s_train)
    rating_scaler.fit(y_train.reshape(-1, 1))

    V_u_train = user_scaler.transform(V_u_train).astype("float32")
    V_u_test = user_scaler.transform(V_u_test).astype("float32")

    V_s_train = song_scaler.transform(V_s_train).astype("float32")
    V_s_test = song_scaler.transform(V_s_test).astype("float32")

    y_train = rating_scaler.transform(y_train.reshape(-1, 1)).astype("float32")
    y_test = rating_scaler.transform(y_test.reshape(-1, 1)).astype("float32")

    # Reshape y for Keras to have shape (num_samples, 1) instead of (num_samples,)
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    return V_u_train, V_u_test, V_s_train, V_s_test, y_train, y_test


def build_model(user_dim, song_dim, embedding_dim=16, learning_rate=0.001):
    """
    Build the content-based recommendation model using TensorFlow Keras.

    Parameters:
        user_dim (int):         Dimension of user feature vectors.
        song_dim (int):         Dimension of song feature vectors.
        embedding_dim (int):    Dimension of the embedding space.
        learning_rate (float):  Learning rate for the optimizer.
    Returns:
        model (tf.keras.Model): Compiled Keras model.
    """
    user_model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', name='user_dense_1'),
        keras.layers.Dense(32, activation='relu', name='user_dense_2'),
        keras.layers.Dense(embedding_dim, name='user_embedding'),
        keras.layers.UnitNormalization(axis=1, name='user_normalization')
    ], name="user_model")
    song_model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', name='song_dense_1'),
        keras.layers.Dense(32, activation='relu', name='song_dense_2'),
        keras.layers.Dense(embedding_dim, name='song_embedding'),
        keras.layers.UnitNormalization(axis=1, name='song_normalization')
    ], name="song_model")

    # Define the inputs for the overall model
    user_input = keras.layers.Input(shape=(user_dim,), name="user_input")
    song_input = keras.layers.Input(shape=(song_dim,), name="song_input")

    # Get the embeddings from the user and song models
    vu = user_model(user_input)
    vs = song_model(song_input)

    # Dot product (cosine similarity due to L2 normalization)
    output = keras.layers.Dot(axes=1, name="dot_product")([vu, vs])

    # Define the overall model
    model = keras.Model(inputs=[user_input, song_input], outputs=output, name="content_based_recommender")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.MeanSquaredError()
    )
    return model


def predict_user_ratings_for_all_songs(model, songs_df, test_user, scalers):
    """
    Predict ratings for all songs for a given test user.

    Parameters:
        model (tf.keras.Model):     Trained recommendation model.
        songs_df (pd.DataFrame):    DataFrame containing song features.
        test_user (dict):           Dictionary of test user features.
        scalers (dict):             Dictionary containing 'user_scaler', 'song_scaler', and 'rating_scaler'.
    Returns:
        song_ids (np.ndarray):      Array of song IDs.
        pred_ratings (np.ndarray):  Array of predicted ratings for the test user.
    """
    # Unpack scalers
    user_scaler = scalers["user_scaler"]
    song_scaler = scalers["song_scaler"]
    rating_scaler = scalers["rating_scaler"]

    # Get and scale all song features
    all_song_features = songs_df[SONG_FEATURE_COLUMNS].to_numpy(dtype=float)
    all_song_features_norm = song_scaler.transform(all_song_features).astype("float32")

    # Get and scale test user features
    user_features = np.array([test_user[g] for g in USER_FEATURE_COLUMNS], dtype=float).reshape(1, -1)
    user_features_norm = user_scaler.transform(user_features).astype("float32")

    # Repeat for all songs to have num_songs vectors
    num_songs = all_song_features_norm.shape[0]
    all_user_features_norm = np.repeat(user_features_norm, repeats=num_songs, axis=0)

    # Predict ratings for the test user for all songs
    scores = model.predict([all_user_features_norm, all_song_features_norm], verbose=0)
    pred_ratings = rating_scaler.inverse_transform(scores).reshape(-1)

    song_ids = songs_df["song_id"].to_numpy()
    return song_ids, pred_ratings


def recommend_top_k(song_ids, pred_ratings, k=10):
    """
    Recommend top-k songs based on predicted ratings.

    Parameters:
        song_ids (np.ndarray):      Array of song IDs.
        pred_ratings (np.ndarray):  Array of predicted ratings.
        k (int):                    Number of top recommendations to return.
    Returns:
        top_song_ids (np.ndarray):   Array of top-k song IDs.
        top_ratings (np.ndarray):    Array of top-k predicted ratings.
        top_indices (np.ndarray):    Indices of the top-k songs in the original array
    """
    song_ids = np.asarray(song_ids).reshape(-1)
    pred_ratings = np.asarray(pred_ratings).reshape(-1)

    k = min(k, len(pred_ratings))
    top_indices = np.argsort(-pred_ratings)[:k]  # descending
    return song_ids[top_indices], pred_ratings[top_indices], top_indices


def main():
    # Load the raw data
    users_df, songs_df, ratings_df = load_raw_data(USERS_DATA_PATH, SONGS_DATA_PATH, RATINGS_DATA_PATH)

    # Process the data into feature matrices
    V_u, V_s, y = process_data(users_df, songs_df, ratings_df, USER_FEATURE_COLUMNS, SONG_FEATURE_COLUMNS)

    # Split the data into training and testing sets
    V_u_train, V_u_test, V_s_train, V_s_test, y_train, y_test = train_test_split(
        V_u, V_s, y,
        test_size=0.2,
        random_state=SEED,
        shuffle=True
    )

    # Define scalers
    scalers = {
        "user_scaler": StandardScaler(),
        "song_scaler": StandardScaler(),
        "rating_scaler": MinMaxScaler(feature_range=(-1, 1))
    }

    # Scale the data
    V_u_train, V_u_test, V_s_train, V_s_test, y_train, y_test = scale_data(
        scalers,
        V_u_train, V_u_test, V_s_train, V_s_test, y_train, y_test
    )

    # Build the model
    model = build_model(
        user_dim=V_u_train.shape[1],
        song_dim=V_s_train.shape[1]
    )

    # Train the model
    model.fit(
        [V_u_train, V_s_train],
        y_train,
        epochs=50,
        shuffle=True,
    )

    # Evaluate the model
    test_loss = model.evaluate([V_u_test, V_s_test], y_test, verbose=0)
    print(f"Final test loss: {test_loss:.4f}")

    # Define a test user for recommendations
    test_user = {
        "rock": 4.5,
        "classical": 0.2,
        "pop": 2.0,
        "jazz": 1.0,
        "hiphop": 3.5
    }

    # Predict ratings for all songs for the test user
    song_ids, pred_ratings = predict_user_ratings_for_all_songs(model, songs_df, test_user, scalers)

    # Recommend top-5 songs
    top_song_ids, top_ratings, top_idx = recommend_top_k(song_ids, pred_ratings, k=5)
    for song_id, rating in zip(top_song_ids, top_ratings):
        print(f"Song ID: {song_id}, Predicted Rating: {rating:.2f}")


if __name__ == "__main__":
    main()

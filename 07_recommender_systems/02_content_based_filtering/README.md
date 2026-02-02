# Content-Based Filtering

This example demonstrates a content-based filtering approach for building a recommender system using explicit
user and item features. The goal is to recommend items (songs) to users based on how well the song content matches the
user’s preferences, without relying on other users’ ratings.

## Model

Each song *i* and each user *j* is represented by a learned latent feature vector (embedding).

Prediction:

ŷ(i, j) = v_u(j) · v_s(i)

where

- v_u(j) = user embedding, learned from user preference features
- v_s(i) = song embedding, learned from song content features

Both embeddings are L2-normalized, making the dot product a measure of directional similarity.

## Cost Function

To measure how well the predicted scores match the true user ratings, a squared error cost function is used,
along with feature scaling to stabilize training.

J = (1 / 2) · Σ (v_u(j) · v_s(i) − y(i,j))²

The objective is to learn embedding functions for users and songs that minimize the overall prediction error.

## Training

The user and song embeddings are learned using neural networks:

- a **user network** maps user preference features to a user embedding
- a **song network** maps song content features to a song embedding

The model is trained using Gradient Descent with the Adam optimizer. TensorFlow automatic differentiation (autodiff) is
used to compute gradients and update model parameters.

The training data consists of user preference features, song content features, and user ratings.

Below are previews of the synthetic training datasets stored as CSV files:

### User Preferences

| user_id | rock | classical | pop | jazz | hiphop |
|--------:|-----:|----------:|----:|-----:|-------:|
|       1 |  4.8 |       0.2 | 1.0 |  0.1 |    2.5 |
|       2 |  0.5 |       4.5 | 0.3 |  3.8 |    0.2 |
|       3 |  2.0 |       1.0 | 4.5 |  0.5 |    3.0 |
|     ... |  ... |       ... | ... |  ... |    ... |

### Song Content

| song_id | rock | classical | pop | jazz | hiphop |
|--------:|-----:|----------:|----:|-----:|-------:|
|       1 |  5.0 |       0.0 | 0.5 |  0.0 |    1.0 |
|       2 |  0.0 |       5.0 | 0.2 |  4.0 |    0.0 |
|       3 |  1.0 |       0.0 | 5.0 |  0.1 |    2.0 |
|     ... |  ... |       ... | ... |  ... |    ... |

### User Ratings

| user_id | song_id | rating |
|--------:|--------:|-------:|
|       1 |       1 |      5 |
|       1 |       2 |      3 |
|       1 |       3 |      0 |
|     ... |     ... |    ... |

## Reproducibility

To ensure reproducible results across runs and between different implementations, a fixed random seed is used throughout
this example.

- NumPy random number generation is seeded
- TensorFlow uses a fixed random seed for weight initialization, data shuffling, and training behavior
- The same seed is applied for data splitting and preprocessing steps

The default seed used in this project is:

```
SEED = 42
```

Changing the seed may lead to slightly different learned parameters and evaluation results.

## Implementation

The implementation uses Numpy and Pandas for loading and preparing the data. For feature scaling and data splitting,
scikit-learn is utilized. For defining, training, and evaluating the neural network model, TensorFlow/Keras is used.

## How to Run

1. Ensure you have Python installed (version 3.6 or higher recommended).
    ```
    python --version
    ```

2. Install the required libraries listed in the requirements.txt file:
    ```bash
    python -m pip install -r requirements.txt
    ```

3. Run the TensorFlow implementation:

```bash
python tensorflow_implementation.py
```

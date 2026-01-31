# Collaborative Filtering

This example demonstrates a collaborative filtering approach for building a recommender system using user-item
interaction data. The goal is to recommend items (songs) to users based on the ratings of users who gave similar ratings
to the same items.

## Model

Each song *i* and each user *j* is represented by a latent feature vector.

Prediction:

ŷ(i, j) = w_j · x_i + b_j

where

- x_i = song features
- w_j = user preferences
- b_j = user bias

## Cost Function

To measure how well the model predictions match the true user ratings, a regularized squared error cost function is
used, along with regularization terms to prevent overfitting.

J = (1 / 2) · Σ R(i,j) · (w_j · x_i + b_j − y(i,j))² + regularization terms

where R(i,j) indicates whether user j rated song i.

## Training

The model parameters (song features, user preferences, and user biases) are learned using Gradient Descent. During
training, the gradients of the cost function with respect to the model parameters are computed and used to iteratively
update the parameters in order to minimize the overall prediction error.

TensorFlow automatic differentiation (autodiff) is used to compute gradients.

Below is a preview of the synthetic training dataset stored as a CSV file:

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
- The same seed is applied for data shuffling and parameter initialization where applicable

The default seed used in this project is:

```
SEED = 42
```

Changing the seed may lead to slightly different learned parameters and evaluation results.

## Implementation

The implementation includes using NumPy for loading data and preprocessing. For optimization and training,
TensorFlow/Keras is used to leverage its efficient computation and automatic differentiation capabilities.

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

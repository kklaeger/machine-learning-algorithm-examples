# Logistic Regression

These examples demonstrate logistic regression using the same used car scenario as
for [linear regression](../01_linear_regression). The goal is to predict whether a customer will buy a car based on
multiple numerical features such as mileage, age, engine power, number of previous owners, and the price of the car.

The algorithm is implemented in two ways:

- A [custom implementation](custom_implementation.py) using NumPy, including the cost function and gradient descent
- A [reference implementation](sklearn_implementation.py) using `scikit-learn` for comparison

## Model

The logistic regression model used in this example follows the form:

    f(x) = σ(w · x + b)

where

    σ(z) = 1 / (1 + exp(-z))

and:

- x is the feature vector (e.g. mileage, age, engine power, number of owners, price)
- w is the weight vector learned during training
- b is the bias (intercept) term
- f(x) = ŷ represents the predicted probability that a car is bought by a customer
- σ(z) is the [sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function)
- y denotes the true target value used during training

## Loss Function

To measure how well the predicted probabilities match the true binary labels, the binary cross-entropy (log loss) is
used.

This loss function penalizes confident but incorrect predictions and is well-suited for binary classification problems.

## Training

The model parameters (weights and bias) are learned using Gradient Descent. During training, the gradients of the loss
function with respect to the model parameters are computed and used to iteratively minimize the binary cross-entropy
loss.

Below is a preview of the synthetic training dataset stored as a CSV file:

| mileage_km | age_years | engine_power_hp | num_previous_owners | price_eur | will_buy |
|-----------:|----------:|----------------:|--------------------:|----------:|---------:|
|     126958 |        15 |             111 |                   0 |      8378 |        1 |
|     151867 |        14 |             327 |                   0 |     21120 |        0 |
|     136932 |         7 |             354 |                   3 |     24160 |        0 |
|     108694 |        13 |             172 |                   2 |     15970 |        1 |
|     124879 |         7 |             160 |                   2 |     16516 |        1 |
|        ... |           |                 |                     |           |          |

## Reproducibility

To ensure reproducible results across runs and between different implementations, a fixed random seed is used throughout
this example.

- NumPy random number generation is seeded
- scikit-learn components use a fixed `random_state`
- The same seed is applied for data shuffling and parameter initialization where applicable

The default seed used in this project is:

```
SEED = 42
```

Changing the seed may lead to slightly different learned parameters and evaluation results.

## Implementation

### Custom Implementation

The custom implementation focuses on clarity and understanding of the algorithm. It includes:

- Explicit computation of the binary cross-entropy (logistic) loss function
- Gradient descent for computing the gradients
- Feature normalization to ensure that all input features have a comparable scale, which improves the stability and
  convergence of gradient descent
- Probability-based predictions using the sigmoid function and a fixed classification threshold

No machine learning libraries are used in this implementation.

### scikit-learn Implementation

The scikit-learn implementation serves as a reference and comparison. It uses:

- `SGDClassifier` for logistic regression trained via stochastic gradient descent using the log loss (binary
  cross-entropy)
- `StandardScaler` for feature normalization

The same training data and feature set are used to allow a direct comparison with the custom implementation.

## How to Run

1. Ensure you have Python installed (version 3.6 or higher recommended).
    ```
   python --version
    ```
2. Install the required libraries listed in the requirements.txt file:
    ```bash
    python -m pip install -r requirements.txt
    ``` 
3. Run the custom implementation:
    ```bash
    python custom_implementation.py
    ```
4. Run the scikit-learn implementation:
    ```bash
    python sklearn_implementation.py
    ```

# Linear Regression

These examples demonstrate linear regression using a used car price prediction scenario. The goal is to predict the
price of a car based on multiple numerical features such as mileage, age, engine power, and number of previous owners.

The algorithm is implemented in two ways:

- A [custom implementation](custom_implementation.py) using NumPy, including the cost function and gradient descent
- A [reference implementation](sklearn_implementation.py) using `scikit-learn` for comparison

## Model

The linear regression model used in this example follows the standard form:

    f(x) = w · x + b

where:

- x is the feature vector (e.g. mileage, age, engine power, number of owners)
- w is the weight vector learned during training
- b is the bias (intercept) term
- f(x) = ŷ is the predicted car price
- y denotes the true target value used during training

## Cost Function

To measure how well the model predictions match the true target values, the Mean Squared Error (MSE) cost function is
used.

The training process aims to minimize the average squared difference between the predicted car prices and the true
prices in the dataset.

## Training

The model parameters (weights and bias) are learned using Gradient Descent. During training, the gradients of the cost
function with respect to the model parameters are computed and used to iteratively update the parameters in order to
minimize the overall prediction error.

Below is a preview of the synthetic training dataset stored as a CSV file:

| mileage_km | age_years | engine_power_hp | num_previous_owners | price_eur |
|-----------:|----------:|----------------:|--------------------:|----------:|
|     126958 |        15 |             111 |                   0 |      8378 |
|     151867 |        14 |             327 |                   0 |     21120 |
|     136932 |         7 |             354 |                   3 |     24160 |
|     108694 |        13 |             172 |                   2 |     15970 |
|     124879 |         7 |             160 |                   2 |     16516 |
|        ... |           |                 |                     |           |

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

- Explicit computation of the Mean Squared Error (MSE) cost function
- Gradient Descent for computing the gradients
- Feature normalization is applied to ensure that all input features have a comparable scale, which improves the
  stability and convergence of gradient descent.
- Simple evaluation using example predictions

No machine learning libraries are used in this implementation.

### scikit-learn Implementation

The scikit-learn implementation serves as a reference and comparison. It uses:

- `SGDRegressor` for linear regression trained via stochastic gradient descent using the squared error loss
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

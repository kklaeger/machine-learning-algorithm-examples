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

## Training Data

Below is a preview of the synthetic training dataset (generated via AI):

| mileage_km | age_years | engine_power_hp | num_previous_owners | price_eur |
|-----------:|----------:|----------------:|--------------------:|----------:|
|     126958 |        15 |             111 |                   0 |      8378 |
|     151867 |        14 |             327 |                   0 |     21120 |
|     136932 |         7 |             354 |                   3 |     24160 |
|     108694 |        13 |             172 |                   2 |     15970 |
|     124879 |         7 |             160 |                   2 |     16516 |
|        ... |           |                 |                     |           |

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

- `SGDRegressor` for linear regression trained via gradient descent
- `StandardScaler` for feature scaling

The same training data and feature set are used to allow a direct comparison with the custom implementation.

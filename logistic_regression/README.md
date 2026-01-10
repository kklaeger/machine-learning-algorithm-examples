# Logistic Regression

These examples demonstrate logistic regression using the same used car scenario as
for [linear regression](../linear_regression). The goal is to predict whether a customer will buy a car based on
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

## Training Data

Below is a preview of the synthetic training dataset (generated via AI):

| mileage_km | age_years | engine_power_hp | num_previous_owners | price_eur | will_buy |
|-----------:|----------:|----------------:|--------------------:|----------:|---------:|
|     126958 |        15 |             111 |                   0 |      8378 |        1 |
|     151867 |        14 |             327 |                   0 |     21120 |        0 |
|     136932 |         7 |             354 |                   3 |     24160 |        0 |
|     108694 |        13 |             172 |                   2 |     15970 |        1 |
|     124879 |         7 |             160 |                   2 |     16516 |        1 |
|        ... |           |                 |                     |           |          |

## Implementation

### Custom Implementation

The custom implementation focuses on clarity and understanding of the algorithm. It includes:

- Explicit computation of the logistic (cross-entropy) loss function
- Gradient descent for computing the gradients
- Feature normalization to ensure that all input features have a comparable scale, which improves the stability and
  convergence of gradient descent
- Probability-based predictions using the sigmoid function and a fixed classification threshold

No machine learning libraries are used in this implementation.

### scikit-learn Implementation

The scikit-learn implementation serves as a reference and comparison. It uses:

- `LogisticRegression` for logistic regression
- `StandardScaler` for feature scaling

The same training data and feature set are used to allow a direct comparison with the custom implementation.

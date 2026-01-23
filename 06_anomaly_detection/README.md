# Anomaly Detection

This example demonstrates anomaly detection using a credit card transaction fraud detection scenario. The goal is to
identify unusual or suspicious transactions by learning what constitutes normal behavior in a dataset of numerical
features.

Anomaly detection is an unsupervised learning task. During training, the model is exposed primarily to normal examples
and learns their statistical properties. New or rare observations that deviate significantly from this learned
distribution are flagged as anomalies.

The algorithm is implemented in two ways:

- A [custom implementation](custom_implementation.py) using NumPy, following the Gaussian anomaly detection
  approach taught in the Coursera Machine Learning Specialization
- A [reference implementation](sklearn_implementation.py) using `scikit-learn` for comparison

## Model

The custom anomaly detection model is based on a multivariate Gaussian distribution. The key steps are:

1. Estimate the mean vector (`mu`) and covariance matrix (`sigma`) from the training data
2. Compute the probability density function `p(x)` for new examples using the multivariate Gaussian formula
3. Select a threshold `epsilon` using a cross-validation set to maximize the F1 score
4. Classify examples as anomalies if their probability `p(x)` is below the threshold

## Training

Anomaly detection training consists of estimating the parameters of the Gaussian distribution using a training dataset
that contains only normal examples.

A separate cross-validation set with labeled anomalies is used exclusively to evaluate the model and to select the
optimal threshold `epsilon`.

Below is a preview of the synthetic training dataset stored as a CSV file:

| amount | hour_of_day | transactions_last_24h | distance_km |
|-------:|------------:|----------------------:|------------:|
|   12.5 |           9 |                     2 |         0.2 |
|   45.6 |          14 |                     2 |         3.4 |
|   60.2 |          16 |                     1 |         5.6 |
|    ... |         ... |                   ... |         ... |

## Feature Scaling

Since the features have different numeric ranges, the data is standardized before training:

- subtract mean (`mu`)
- divide by standard deviation (`sigma`)

Scaling parameters are computed on the training set and reused for cross-validation and inference. This ensures that
each feature contributes comparably to the probability computation and prevents data leakage.

## Reproducibility

To ensure reproducible results across runs and between different implementations, a fixed random seed is used throughout
this example.

- NumPy random number generation is seeded
- scikit-learn components use a fixed `random_state`

The default seed used in this project is:

```
SEED = 42
```

Changing the seed may lead to slightly different detected anomalies, especially for stochastic models such as Isolation
Forest.

## Implementation

### Custom Implementation

The custom implementation includes:

- Feature scaling using mean and standard deviation
- Estimation of multivariate Gaussian parameters
- Probability density computation
- Threshold selection using F1 score
- Explicit anomaly prediction logic

No machine learning libraries are used in this implementation.

### scikit-learn Implementation

The scikit-learn implementation serves as a reference and uses:

- `StandardScaler` for feature normalization
- `IsolationForest` for anomaly detection

The same dataset and preprocessing pipeline are used to allow a direct comparison with the custom implementation.

## How to Run

1. Ensure you have Python installed (version 3.6 or higher recommended).
    ```bash
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

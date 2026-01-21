# K-Means Clustering

This example demonstrates K-Means clustering using a simple customer segmentation scenario. The goal is to group
customers into segments without any labels based on their similarity in a small set of numerical features.

The algorithm is implemented in two ways:

- A [custom implementation](custom_implementation.py) using NumPy, following the K-Means steps explicitly
- A [reference implementation](sklearn_implementation.py) using `scikit-learn` for comparison

## Model

K-Means is an unsupervised learning algorithm that partitions the dataset into K clusters. Each cluster is represented
by a centroid (the mean of all points assigned to that cluster).

The algorithm iterates the following steps until convergence:

1. Initialize: K centroids (randomly chosen from the data)
2. Assign: each data point to its closest centroid (Euclidean distance)
3. Update: each centroid by computing the mean of the assigned points

The output of K-Means is:

- `idx`: the cluster assignment for each data point
- `centroids`: the learned cluster centers

## Training

K-Means training involves repeatedly performing the assignment and update steps until the centroids no longer change
significantly (convergence). The quality of the clustering can be evaluated using the inertia metric, which measures the
sum of squared distances between data points and their assigned centroids. Lower inertia indicates better clustering.

Below is a preview of the synthetic training dataset stored as a CSV file:

| annual_income | spending_score | age |
|--------------:|---------------:|----:|
|         25000 |             20 |  45 |
|         28000 |             25 |  52 |
|         30000 |             18 |  48 |
|         35000 |             22 |  55 |
|         32000 |             15 |  50 |
|           ... |            ... | ... |

In this example, the number of clusters is set to `K = 4`

This choice is made for simplicity and interpretability in the customer segmentation scenario. The learned clusters can
be interpreted as typical customer groups such as:

- Budget / price-sensitive customers (low income, low spending)
- Impulse buyers (low income, high spending)
- Savers (high income, low spending)
- Premium / VIP customers (high income, high spending)

## Feature Scaling

K-Means uses Euclidean distance and is therefore highly sensitive to feature scales. Since `annual_income` has a much
larger numeric range than the other features, the dataset is standardized before clustering:

- subtract mean (`mu`)
- divide by standard deviation (`sigma`)

This ensures that each feature contributes comparably to the distance computation.

## Reproducibility

To ensure reproducible results across runs and between different implementations, a fixed random seed is used throughout
this example.

- NumPy random number generation is seeded
- scikit-learn components use a fixed `random_state`
- The same seed is applied for centroid initialization where applicable

The default seed used in this project is:

```
SEED = 42
```

Changing the seed may lead to different initial centroids and slightly different final clusters.

## Implementation

### Custom Implementation

The custom implementation follows the course-style K-Means steps explicitly:

- Feature scaling via mean and standard deviation
- Random centroid initialization from the dataset
- Repeated assignment and centroid update steps
- A convergence check based on centroid movement
- Inertia computation for evaluation

No machine learning libraries are used in this implementation.

### scikit-learn Implementation

The scikit-learn implementation serves as a reference and comparison. It uses:

- `KMeans` for clustering
- `StandardScaler` for feature normalization

The same dataset and preprocessing steps are used to allow a direct comparison with the custom implementation.

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

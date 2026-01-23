import numpy as np
from pathlib import Path

from utils.data_utils import load_data

SEED = 42

# Ensure reproducible results
np.random.seed(SEED)

np.set_printoptions(precision=2, suppress=True)

BASE_DIR = Path(__file__).resolve().parent
TRAINING_DATA_PATH = BASE_DIR / "data" / "training_data.csv"


def scale_features(X, mu=None, sigma=None):
    """
    Scales the features using mean and standard deviation.

    If mu and sigma are None, they are computed from X and returned.
    If mu and sigma are provided, only the scaled features are returned.

    Parameters:
        X (np.ndarray):     Feature matrix.
        mu (np.ndarray):    Mean values for each feature. If None, compute from X.
        sigma (np.ndarray): Standard deviation for each feature. If None, compute from X.

    Returns:
        X_scaled (np.ndarray): Scaled feature matrix.
        mu (np.ndarray):      Mean values used for scaling.
        sigma (np.ndarray):   Standard deviation values used for scaling.
    """
    compute_params = (mu is None) or (sigma is None)

    if compute_params:
        mu = np.mean(X, axis=0)
        sigma = np.std(X, axis=0)

    # Avoid division by zero for constant features
    sigma = np.where(sigma == 0, 1.0, sigma)
    X_scaled = (X - mu) / sigma

    if compute_params:
        return X_scaled, mu, sigma
    else:
        return X_scaled


def inverse_scale_features(X_scaled, mu, sigma):
    """
    Inverse transforms the scaled features back to the original scale.

    Parameters:
        X_scaled (np.ndarray): Scaled feature matrix.
        mu (np.ndarray):       Mean values used for scaling.
        sigma (np.ndarray):    Standard deviation values used for scaling.

    Returns:
        X_original (np.ndarray): Feature matrix in the original scale.
    """
    return X_scaled * sigma + mu


def initialize_centroids(X, K):
    """
    Randomly initialize K centroids from the dataset X.
    Parameters:
        X (np.ndarray): Feature matrix.
        K (int):        Number of clusters.
    Returns:
        centroids (np.ndarray): Initialized centroids.
    """
    if K > X.shape[0]:
        raise ValueError("K must be less than or equal to number of data points.")

    m = X.shape[0]
    random_indices = np.random.choice(m, K, replace=False)
    centroids = X[random_indices].copy()
    return centroids


def find_closest_centroids(X, centroids):
    """
    Finds the closest centroid for each data point in X.

    Parameters:
        X (np.ndarray):         Feature matrix.
        centroids (np.ndarray): Centroid matrix.
    Returns:
        idx (np.ndarray):       Array of centroid assignments for each data point.
    """
    idx = np.zeros(X.shape[0], dtype=int)

    for i in range(X.shape[0]):
        distances = np.zeros(len(centroids))
        for j, centroid in enumerate(centroids):
            distances[j] = np.linalg.norm(X[i] - centroid)
        idx[i] = np.argmin(distances)
    return idx


def update_centroids(X, idx, K):
    """
    Updates centroids by computing the mean of all data points assigned to each centroid.

    Parameters:
        X (np.ndarray):     Feature matrix.
        idx (np.ndarray):   Array of centroid assignments for each data point.
        K (int):            Number of clusters.

    Returns:
        centroids (np.ndarray): Updated centroids.
    """
    n = X.shape[1]
    centroids = np.zeros((K, n))

    for i in range(K):
        points_assigned = X[idx == i]
        if len(points_assigned) > 0:
            centroids[i] = np.mean(points_assigned, axis=0)
        else:
            # If no points are assigned to the centroid, reinitialize it randomly
            centroids[i] = X[np.random.choice(X.shape[0])].copy()

    return centroids


def k_means(X, K, max_iters=100, tol=1e-4, inits=1):
    """
    Runs the K-Means algorithm on the dataset X. Each row of X is a single data point.
    Parameters:
        X (np.ndarray):     Feature matrix.
        K (int):            Number of clusters.
        max_iters (int):    Maximum number of iterations for each initialization.
        tol (float):        Tolerance to declare convergence.
        inits (int):        Number of random initializations.
    Returns:
        best_centroids (np.ndarray): Centroids from the best initialization.
    """
    best_centroids = None
    best_inertia = float('inf')

    for init in range(inits):
        centroids = initialize_centroids(X, K)

        for iteration in range(max_iters):
            idx = find_closest_centroids(X, centroids)
            new_centroids = update_centroids(X, idx, K)

            # Check for convergence
            if np.linalg.norm(new_centroids - centroids) < tol:
                break

            centroids = new_centroids

        # Ensure idx matches final centroids (and avoid uninitialized idx edge-case)
        idx = find_closest_centroids(X, centroids)
        inertia = compute_inertia(X, centroids, idx)

        if inertia < best_inertia:
            best_inertia = inertia
            best_centroids = centroids

    return best_centroids, best_inertia


def compute_inertia(X, centroids, idx):
    """
    Computes the inertia (sum of squared distances to the closest centroid).

    Parameters:
        X (np.ndarray):         Feature matrix.
        centroids (np.ndarray): Centroid matrix.
        idx (np.ndarray):       Array of centroid assignments for each data point.

    Returns:
        inertia (float): Sum of squared distances to the closest centroid.
    """
    diffs = X - centroids[idx]
    return np.sum(diffs ** 2)


def main():
    # Load the training data from the CSV file
    X = load_data(TRAINING_DATA_PATH, with_target=False)

    # Scale features
    X_scaled, mu, sigma = scale_features(X)

    # Define number of clusters and number of initializations
    K = 4
    inits = 10

    # Train the K-Means Model
    centroids, inertia = k_means(
        X=X_scaled,
        K=K,
        max_iters=100,
        tol=1e-4,
        inits=inits
    )

    print("\nEvaluation:")
    print(f"Train inertia: {inertia:.2f}")

    # Inverse-transform centroids for interpretability
    centers_original = inverse_scale_features(centroids, mu, sigma)
    print("\nCluster centers (original scale):")
    print("annual_income", "spending_score", "age")
    print(centers_original)
    # Cluster centers (original scale):
    #  annual_income  spending_score    age
    # [35100            79.6            26.9] -> Cluster 0: Impulse
    # [85100            79.6            37.5] -> Cluster 1: Premium
    # [35100            23.8            49.2] -> Cluster 2: Budget
    # [85100            25.7            59.6] -> Cluster 3: Savers

    # Predict cluster assignments for new customers
    test_customers = np.array([
        [22000, 18, 58],  # low income, low spending, older -> Cluster 2 (Budget)
        [30000, 85, 24],  # low income, high spending, young -> Cluster 0 (Impulse)
        [92000, 82, 39],  # high income, high spending, middle-aged -> Cluster 1 (Premium)
    ], dtype=float)

    # Use same mu/sigma to transform test rows (returns only scaled array)
    test_customers_scaled = scale_features(test_customers, mu, sigma)
    print("\nScaled test customers:")
    print(test_customers_scaled)

    # predict needs both X and centroids
    clusters = find_closest_centroids(test_customers_scaled, centroids)

    print("\nCluster assignments for new customers:")
    print("Customer 1 -> Cluster", clusters[0])  # Cluster 2
    print("Customer 2 -> Cluster", clusters[1])  # Cluster 0
    print("Customer 3 -> Cluster", clusters[2])  # Cluster 1


if __name__ == "__main__":
    main()

import numpy as np
from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from utils.data_utils import load_data

SEED = 42

# Ensure reproducible results
np.random.seed(SEED)

np.set_printoptions(precision=2, suppress=True)

BASE_DIR = Path(__file__).resolve().parent
TRAINING_DATA_PATH = BASE_DIR / "data" / "training_data.csv"


def main():
    # Load the training data from the CSV file and
    X = load_data(TRAINING_DATA_PATH, with_target=False)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Define number of clusters and number of initializations
    K = 4
    inits = 10

    # K-Means Clustering
    model = KMeans(
        n_clusters=K,
        init='k-means++',
        n_init=inits,
        random_state=SEED
    )

    # Train the model
    model.fit(X_scaled)

    print("\nEvaluation:")
    print(f"Train inertia: {model.inertia_:.2f}")

    centers_original = scaler.inverse_transform(model.cluster_centers_)
    print("\nCluster centers (original scale):")
    print("annual_income", "spending_score", "age")
    print(centers_original)
    # Cluster centers (original scale):
    #  annual_income  spending_score    age
    # [35100            79.6            26.9] -> Cluster 0: Impulse
    # [85100            25.7            59.6] -> Cluster 1: Savers
    # [35100            23.8            49.2] -> Cluster 2: Budget
    # [85100            79.6            37.5] -> Cluster 3: Premium

    # Predict cluster assignments for new customers
    test_customers = np.array([
        [22000, 18, 58],  # low income, low spending, older -> Cluster 2 (Budget)
        [30000, 85, 24],  # low income, high spending, young -> Cluster 0 (Impulse)
        [92000, 82, 39],  # high income, high spending, middle-aged -> Cluster 3 (Premium)
    ], dtype=float)

    test_customers_scaled = scaler.transform(test_customers)
    clusters = model.predict(test_customers_scaled)

    print("\nCluster assignments for new customers:")
    print("Customer 1 -> Cluster", clusters[0])  # Cluster 2
    print("Customer 2 -> Cluster", clusters[1])  # Cluster 0
    print("Customer 3 -> Cluster", clusters[2])  # Cluster 3


if __name__ == "__main__":
    main()

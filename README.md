# Machine Learning Algorithm Examples

This repository contains simple examples of machine learning algorithms implemented for testing, experimentation, and
demonstration purposes.

The goal is to provide simple implementations that illustrate the core ideas behind common machine learning techniques.

For some algorithms, a custom implementation is provided for learning purposes, alongside a reference implementation
using established libraries such as scikit-learn or TensorFlow. The custom implementations are not optimized for
production use. Their purpose is clarity and understanding, not performance.

## Overview

The following table summarizes the algorithms included in this repository:

| #  | Algorithm           | Learning Type | Task Type         | Custom Implementation | Reference Implementation |
|----|---------------------|---------------|-------------------|-----------------------|--------------------------|
| 01 | Linear Regression   | Supervised    | Regression        | Yes                   | scikit-learn             |
| 02 | Logistic Regression | Supervised    | Classification    | Yes                   | scikit-learn             |
| 03 | Decision Trees      | Supervised    | Classification    | Yes                   | scikit-learn             |
| 04 | Neural Networks     | Supervised    | Classification    | Yes                   | TensorFlow               |
| 05 | K-Means Clustering  | Unsupervised  | Clustering        | Yes                   | scikit-learn             |
| 06 | Anomaly Detection   | Unsupervised  | Anomaly Detection | Yes                   | scikit-learn             |
| 07 | Recommender Systems | Supervised    | Recommendation    | No                    | TensorFlow               | 

## Algorithms

### [Linear Regression](01_linear_regression)

Linear regression is used to predict continuous numerical values.

In this repository, both a custom implementation and a scikit-learn reference implementation are provided.

**Example:** Predicting used car prices based on features like mileage, age, and brand.

### [Logistic Regression](02_logistic_regression)

Logistic regression is a binary classification algorithm used to predict probabilities.

In this repository, both a custom implementation and a scikit-learn reference implementation are provided.

**Example:** Predicting whether a customer will buy a used car based on features like price, mileage, age, and brand.

### [Decision Trees](03_decision_trees)

Decision trees are versatile models that can be used for both classification and regression tasks.

**Example:** Predicting the diabetes risk level of patients based on health metrics like BMI, age, blood pressure, and
glucose levels.

### [Neural Networks](04_neural_networks)

Neural networks are flexible models composed of multiple layers that can learn complex, non-linear relationships in
data. This repository covers both dense and convolutional neural networks.

#### Dense Neural Networks

Dense neural networks consist of fully connected layers and are commonly used for learning non-linear patterns in
tabular data.

In this repository, two examples of dense networks are implemented in a custom way as well as using TensorFlow:

- **Binary Classification Example:** Predicting whether a customer will buy a used car based on features like price,
  mileage, age, and brand.
- **Multiclass Classification Example:** Predicting the price category of a used car (e.g. cheap, average, expensive).

#### Convolutional Neural Networks (CNNs)

Convolutional neural networks are specialized neural networks designed for processing image data. They use convolutional
layers to automatically learn spatial features such as edges and shapes.

In this repository, two CNN examples are provided using TensorFlow:

- **Binary Classification Example:** Classifying images of cats vs. dogs.
- **Multiclass Classification Example:** TBD

Additional algorithms may be added over time following the same structure and design principles.

### [K-Means Clustering](05_k_means)

K-Means clustering is an unsupervised learning algorithm used to group similar data points into clusters based on
feature similarity.

In this repository, both a custom implementation and a scikit-learn reference implementation are provided.

**Example:** Segmenting customers into distinct groups based on purchasing behavior and demographics.

### [Anomaly Detection](06_anomaly_detection)

Anomaly detection is an unsupervised learning technique used to identify unusual or rare events in data that deviate
from normal patterns.

In this repository, both a custom implementation and a scikit-learn reference implementation are provided.

**Example:** Detecting fraudulent credit card transactions based on transaction amount, time, location, and other
features.

### [Recommender Systems](07_recommender_systems)

Recommender systems are algorithms designed to suggest items to users based on their preferences and behavior.

#### Collaborative Filtering

Collaborative filtering is a popular approach for building recommender systems using user-item interaction data. The
goal is to recommend items (e.g., songs, movies, products) to users based on the ratings or interactions of users who
gave similar ratings to the same items.

**Example:** Recommending songs to users based on their past ratings and the ratings of other users.

#### Content-Based Filtering

Content-based filtering is an approach for building recommender systems using explicit user and item features. The goal
is to recommend items (e.g., songs, movies, products) to users based on how well the item content matches the user's
preferences, without relying on other users' ratings.

**Example:** Recommending songs to users based on song features (e.g., genre, tempo) and user preferences (e.g., age,
favorite genres).

## Notes

All examples in this repository use a fixed random seed (default: 42) to ensure reproducible results across runs and
implementations.

## How to Use

Python 3.x is required along with the libraries listed in the requirements.txt file.

Each algorithm has its own directory with implementation files and example scripts. To run an example, navigate to the
corresponding directory and execute the example script.

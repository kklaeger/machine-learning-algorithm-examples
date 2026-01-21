# Machine Learning Algorithm Examples

This repository contains simple examples of machine learning algorithms implemented for testing, experimentation, and
demonstration purposes.

The goal is to provide simple implementations that illustrate the core ideas behind common machine learning techniques.

For some algorithms, a custom implementation is provided for learning purposes, alongside a reference implementation
using established libraries such as scikit-learn or TensorFlow. The custom implementations are not optimized for
production use. Their purpose is clarity and understanding, not performance.

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

## Notes

All examples in this repository use a fixed random seed (default: 42) to ensure reproducible results across runs and
implementations.

## How to Use

Python 3.x is required along with the libraries listed in the requirements.txt file.

Each algorithm has its own directory with implementation files and example scripts. To run an example, navigate to the
corresponding directory and execute the example script.

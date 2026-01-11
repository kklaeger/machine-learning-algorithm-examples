# Machine Learning Algorithm Examples

This repository contains simple examples of machine learning algorithms implemented for testing, experimentation, and
demonstration purposes.

The goal is to provide simple implementations that illustrate the core ideas behind common machine learning techniques.

## Algorithms

- [Linear Regression](linear_regression)
    - Custom implementation using NumPy
    - Reference implementation using scikit-learn
    - Linear regression example based on car purchase decisions

- [Logistic Regression](logistic_regression)
    - Custom implementation using NumPy
    - Reference implementation using scikit-learn
    - Binary classification example based on car purchase decisions

- [Neural Networks](neural_networks)
    - Dense neural networks using a custom implementation (NumPy) and a reference implementation (TensorFlow)
    - Convolutional neural networks (CNNs) for image classification using TensorFlow
    - Examples ordered by increasing complexity

Additional algorithms may be added over time following the same structure and design principles.

## Run

### Requirements

Python 3 and the following dependencies are required:

- numpy
- scikit-learn

```bash
python3 -m pip install -r requirements.txt
```

### Run an Example

For example, to run the linear regression example:

```bash
python3 linear-regression/custom_implementation.py
```

To run the scikit-learn reference implementation:

```bash
python3 linear-regression/sklearn_implementation.py
```

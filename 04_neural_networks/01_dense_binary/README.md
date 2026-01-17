# Dense Neural Network – Binary Classification

This example demonstrates binary classification using a dense (fully connected) neural network. It is based on the same
used car purchase scenario as the [logistic regression example](../../02_logistic_regression). The goal is to predict
whether a customer will buy a car based on multiple numerical features such as mileage, age, engine power, number of
previous owners, and the price of the car.

The neural network is implemented in two ways:

- A [custom implementation](custom_implementation.py) using NumPy, including forward propagation, backpropagation, and
  gradient-based optimization
- A [reference implementation](tensorflow_implementation.py) using TensorFlow/Keras for comparison

## Architecture

The dense neural network used in this example has the following structure:

- Input layer corresponding to the numerical feature vector
- One hidden dense layer with ReLU activation
- Output layer with a sigmoid activation function

Compared to logistic regression, the additional hidden layer enables the model to learn non-linear decision boundaries.

## Loss Function

Binary cross-entropy is used as the loss function, which is well-suited for binary classification problems.

## Training

The network is trained using gradient-based optimization. Gradients are computed via backpropagation and used to update
the model parameters.

The training data is identical to the dataset used in the logistic regression example. A preview of the synthetic
dataset is shown below:

| mileage_km | age_years | engine_power_hp | num_previous_owners | price_eur | will_buy |
|-----------:|----------:|----------------:|--------------------:|----------:|---------:|
|     126958 |        15 |             111 |                   0 |      8378 |        1 |
|     151867 |        14 |             327 |                   0 |     21120 |        0 |
|     136932 |         7 |             354 |                   3 |     24160 |        0 |
|     108694 |        13 |             172 |                   2 |     15970 |        1 |
|     124879 |         7 |             160 |                   2 |     16516 |        1 |
|        ... |           |                 |                     |           |          |

## Implementations

### Custom Implementation

A NumPy-based implementation of a dense neural network built from scratch. This implementation focuses on understanding
forward propagation, backpropagation, activation functions, and gradient-based optimization.

### TensorFlow Implementation

A reference implementation using TensorFlow/Keras that solves the same problem using a high-level deep learning
framework.

## Notes

- Feature scaling is applied to ensure stable and efficient training of the neural network.
- The custom implementation avoids unnecessary abstractions to keep the training process transparent.

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
4. Run the TensorFlow implementation:
    ```bash
    python tensorflow_implementation.py
    ```

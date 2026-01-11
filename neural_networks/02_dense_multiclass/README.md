# Dense Neural Network – Multiclass Classification

This example demonstrates multiclass classification using a dense (fully connected) neural network. The goal is to
predict the price category of a car (cheap, average, expensive) based on its features.

The neural network is implemented in two ways:

- A [custom implementation](custom_implementation.py) using NumPy, including forward propagation, backpropagation, and
  gradient-based optimization
- A [reference implementation](tensorflow_implementation.py) using TensorFlow/Keras for comparison

## Architecture

The dense neural network used in this example consists of:

- An input layer corresponding to the numerical feature vector
- One hidden dense layer with non-linear ReLU activation function
- An output layer with linear activation and three output units (one per class)
- Softmax applied outside the model to obtain class probabilities
- Categorical cross-entropy as the loss function
- Gradient-based optimization

---

## Training Data

The training data is identical to the dataset used in the logistic regression example, except that the target is now a
multiclass label.

The dataset contains the following columns:

- `mileage_km`
- `age_years`
- `engine_power_hp`
- `num_previous_owners`
- `price_category` (integer labels: `0 = cheap`, `1 = average`, `2 = expensive`)

Below is a preview of the synthetic training dataset (generated via AI):

| mileage_km | age_years | engine_power_hp | num_previous_owners | price_category |
|-----------:|----------:|----------------:|--------------------:|---------------:|
|     126958 |        15 |             111 |                   0 |              0 |
|     151867 |        14 |             327 |                   0 |              1 |
|     136932 |         7 |             354 |                   3 |              2 |
|     108694 |        13 |             172 |                   2 |              0 |
|     124879 |         7 |             160 |                   2 |              1 |
|        ... |           |                 |                     |                |

---

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

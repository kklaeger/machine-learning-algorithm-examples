# Neural Networks

This directory contains neural network implementations using TensorFlow as well as custom implementations based on
NumPy. The focus is on understanding the fundamentals of neural networks and their practical application.

The dense neural network examples build upon concepts introduced in logistic regression and extend them using multiple
layers and non-linear transformations.

The examples cover both **dense (fully connected)** and **convolutional** neural networks.

## Dense Neural Networks

Dense neural networks, also known as multilayer perceptrons (MLPs), consist of fully connected layers and are commonly
used for learning non-linear patterns in tabular data.

These examples are used to experiment core neural network concepts such as:

- Forward propagation
- Backpropagation
- Activation functions
- Loss functions
- Optimization using gradient-based methods

For this reason, the dense network examples include both custom NumPy-based implementations (for learning and clarity)
and TensorFlow reference implementations.

## Convolutional Neural Networks

Convolutional neural networks (CNNs) are introduced for more complex tasks, primarily image classification, where
spatial structure in the data is important.

CNN examples focus on:

- Learning spatial features automatically
- Separating data pipelines, model definition, training, and evaluation
- Practical usage of TensorFlow/Keras for image-based tasks

Because of the complexity of CNNs, these examples are implemented using TensorFlow only.

## Projects

- [01_dense_binary](01_dense_binary)
    - Binary classification using a dense neural network (MLP).
    - Custom NumPy implementation and TensorFlow reference implementation.

- [02_dense_multiclass](02_dense_multiclass)
    - Multiclass classification using a dense neural network.
    - Custom NumPy implementation and TensorFlow reference implementation.

- [03_cnn_binary](03_cnn_binary)
    - Binary image classification using a convolutional neural network (CNN).
    - Implemented using TensorFlow/Keras.

- [04_cnn_multiclass](04_cnn_multiclass)
    - TBD

Additional neural network examples may be added over time following the same structure and learning-oriented design
principles.

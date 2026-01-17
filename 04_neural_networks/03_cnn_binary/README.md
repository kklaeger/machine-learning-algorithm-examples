# Convolutional Neural Networks - Binary Classification

This example demonstrates binary classification using a convolutional neural network (CNN). The goal is to predict
whether an input image contains a cat or a dog. Unlike previous examples using tabular data, this model learns features
directly from raw image pixels using convolutional layers.

## Dataset

The project uses the
[Cats vs Dogs dataset from tensorflow_datasets (tfds)](https://www.tensorflow.org/datasets/catalog/cats_vs_dogs) that
consists of labeled images of cats and dogs.

The dataset is split as follows:

* **Training:** 80%
* **Validation:** 10%
* **Test:** 10%

## Directory Structure

The project is organized into the following main components:

- Data Pipeline (`data.py`)
- Model Architecture (`model.py`)
- Training (`train.py`)
- Prediction & Visualization (`predict.py`)
- Evaluation (`evaluate.py`)

### Data Pipeline (`data.py`)

The data pipeline loads the raw images from TFDS, preprocesses them, and builds efficient `tf.data.Dataset` pipelines
for training, validation, and testing.

Because the images have varying original resolutions, they are resized with preserved aspect ratio and padding so that
all model inputs have the same shape.

Preprocessing ensures that all images entering the CNN have shape (160, 160, 3) and pixel values normalized to
the [0, 1] range.

### Model Architecture (`model.py`)

The CNN consists of the following components:

- An input layer for images of shape (160, 160, 3)
- Data augmentation layers for random transformations during training
- Three convolutional blocks with ReLU activations and max pooling
- A global average pooling layer to reduce spatial dimensions
- A fully connected classifier head with dropout for regularization
- A sigmoid output layer for binary classification

High-level architecture:

```
Input (160, 160, 3)
 → Data Augmentation
 → Conv(32) → MaxPool
 → Conv(64) → MaxPool
 → Conv(128) → MaxPool
 → GlobalAveragePooling
 → Dense(128) + ReLU
 → Dropout(0.3)
 → Output: Dense(1) + Sigmoid
```

The output layer produces a single probability value indicating the likelihood of the image being a dog.

### Training (`train.py`)

The model is trained using the Adam optimizer and binary cross-entropy loss, which are well-suited for binary image
classification tasks.

Training uses the following best practices:

- Early stopping based on validation loss to prevent overfitting
- Model checkpointing to save the best model during training
- Reproducible results via random seeds

The best model is saved to `models/model.keras`.

This trained model is later used for prediction and evaluation.

### Prediction & Visualization (`predict.py`)

This script loads the trained model and performs inference on random samples from the test set. It prints the true
label, predicted label, and predicted probability for each sample.

Example output:

```
Sample 1: True=cat, Pred=cat, Prob=0.23
Sample 2: True=dog, Pred=cat, Prob=0.41
```

### Evaluation (`evaluate.py`)

Evaluation is performed on the full test set and includes test loss and accuracy metrics, a confusion matrix, and
class-wise precision and recall scores.

## Reproducibility

To ensure reproducible results across runs and between different implementations, a fixed random seed is used throughout
this example.

- NumPy random number generation is seeded
- TensorFlow uses a fixed random seed for weight initialization, data shuffling, and training behavior
- The same seed is applied for data shuffling and parameter initialization where applicable

The default seed used in this project is:

```
SEED = 42
```

Changing the seed may lead to slightly different learned parameters and evaluation results.

## Notes

The focus of this example is on understanding the CNN pipeline and architecture, not on achieving state-of-the-art
accuracy.

## How to Run

1. Ensure you have Python installed (version 3.6 or higher recommended).
    ```
   python --version
    ```
2. Install the required libraries listed in the requirements.txt file:
    ```bash
    python -m pip install -r requirements.txt
    ``` 
3. Train the CNN model by running:
    ```bash
    python train.py
    ```
4. After training, you can use the trained model for predictions by running:
    ```bash
    
    python predict.py
    ```
5. Finally, evaluate the model's performance on the test set by running:
    ```bash
    python evaluate.py
    ```

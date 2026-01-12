# Convolutional Neural Networks - Binary Classification

This example demonstrates binary classification using a convolutional neural network (CNN). The goal is to predict 
whether an input image contains a cat or a dog.

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
- Binary cross-entropy loss and Adam optimizer

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

---

### Training (`train.py`)

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

---

### Evaluation (`evaluate.py`)

Evaluation is performed on the full test set and includes test loss and accuracy metrics, a confusion matrix, and
precision & recall calculations.

## Run

To train the CNN model, run:

```bash
python3 train.py
```

To make predictions on random test samples, run:

```bash

python3 predict.py
```

To evaluate the trained model, run:

```bash
python3 evaluate.py
```

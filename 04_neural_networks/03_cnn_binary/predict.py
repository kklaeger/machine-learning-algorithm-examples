import tensorflow as tf
import matplotlib.pyplot as plt

from data import load_raw_data, prepare_dataset

MODEL_PATH = "models/model.keras"
SEED = 42

def get_random_examples(dataset, number_of_examples):
    """
    Retrieves a specified number of random examples from the dataset.
    Parameters:
        - dataset (tf.data.Dataset):    The input dataset.
        - number_of_examples (int):     Number of random examples to retrieve.

    Returns:
        - images (tf.Tensor): A batch of randomly selected images.
        - labels (tf.Tensor): Corresponding labels for the selected images.
    """
    dataset = dataset.shuffle(1000, seed=SEED, reshuffle_each_iteration=True).take(number_of_examples)

    images, labels = [], []
    for image, label in dataset:
        images.append(image)
        labels.append(label)

    images = tf.stack(images, axis=0)
    labels = tf.stack(labels, axis=0)

    return images, labels


def predict(model, images):
    """
    Predicts the class probabilities for a batch of images using the provided model.

    Parameters:
        - model (tf.keras.Model):   The trained CNN model.
        - images (tf.Tensor):       A batch of images to predict.

    Returns:
        - probabilities (np.ndarray): Array of predicted probabilities for the positive class.
    """
    probabilities = model.predict(images, verbose=0).reshape(-1)
    return probabilities


def visualize_predictions(images, labels, probabilities, class_names, threshold=0.5):
    """
    Visualizes the predictions made by the model on a batch of images.

    Parameters:
        - images (tf.Tensor):           A batch of images.
        - labels (tf.Tensor):           Corresponding true labels for the images.
        - probabilities (np.ndarray):   Predicted probabilities for the positive class.
        - class_names (list):           List of class names corresponding to label indices.
        - threshold (float):            Probability threshold for classifying as positive class (default is 0.5).
    """
    number_of_samples = images.shape[0]

    # Give the figure more vertical space
    plt.figure(figsize=(number_of_samples * 4, 5))

    for i in range(number_of_samples):
        true_name = class_names[int(labels[i].numpy())]
        pred_name = class_names[int(probabilities[i] >= threshold)]

        print(
            f"Sample {i + 1}: "
            f"True={true_name}, "
            f"Pred={pred_name}, "
            f"Prob={probabilities[i]:.3f}"
        )

        plt.subplot(1, number_of_samples, i + 1)
        plt.imshow(images[i].numpy())

        correct = (pred_name == true_name)

        symbol = "✔" if correct else "✘"
        color = "green" if correct else "red"

        plt.title(f"{symbol} Prediction: {pred_name}", color=color)
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def main():
    # Load the raw dataset
    _, _, test_data_raw, data_info = load_raw_data()

    # Prepare test dataset
    test_data = prepare_dataset(test_data_raw, shuffle=False, do_batch=False)

    print(f"Dataset: {data_info.name}")
    print(f"Number of test samples: {test_data_raw.cardinality().numpy()}")

    # Get random examples from the test dataset
    images, labels = get_random_examples(test_data, number_of_examples=8)

    # Load the trained model
    model = tf.keras.models.load_model(MODEL_PATH)

    # Predict probabilities for the selected images
    probabilities = predict(
        model=model,
        images=images
    )

    # Visualize the predictions
    visualize_predictions(
        images=images,
        labels=labels,
        probabilities=probabilities,
        class_names=data_info.features["label"].names,
        threshold=0.5
    )


if __name__ == "__main__":
    main()

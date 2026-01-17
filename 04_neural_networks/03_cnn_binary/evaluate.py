import tensorflow as tf

from data import load_raw_data, prepare_dataset
from train import MODEL_PATH

BATCH_SIZE = 32
THRESHOLD = 0.5


def evaluate(model, test_data, class_names, threshold=0.5):
    """
    Evaluates the model on the test dataset and computes performance metrics including loss, accuracy,
    confusion matrix, precision, and recall.

    Parameters:
        - model (tf.keras.Model):       The trained CNN model to be evaluated.
        - test_data (tf.data.Dataset):  The test dataset for evaluation.
        - class_names (list):           List of class names corresponding to label indices.
        - threshold (float):            Probability threshold for classifying as positive class (default is 0.5).
    """
    # Evaluate the model on the test dataset
    loss, acc = model.evaluate(test_data, verbose=0)
    print(f"Test loss:     {loss:.4f}")
    print(f"Test accuracy: {acc:.4f}")

    # Compute a confusion matrix
    y_true = []
    y_predicted = []
    for images, labels in test_data:
        probabilities = model.predict(images, verbose=0).reshape(-1)
        predictions = (probabilities >= threshold).astype("int32")

        y_true.append(labels.numpy().astype("int32"))
        y_predicted.append(predictions)

    y_true = tf.concat([tf.constant(x) for x in y_true], axis=0)
    y_predicted = tf.concat([tf.constant(x) for x in y_predicted], axis=0)

    cm = tf.math.confusion_matrix(y_true, y_predicted, num_classes=2).numpy()
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

    print("\nConfusion Matrix (rows=true, cols=pred):")
    print(f"           Pred {class_names[0]:>5}   Pred {class_names[1]:>5}")
    print(f"True {class_names[0]:>5}     {tn:>6}       {fp:>6}")
    print(f"True {class_names[1]:>5}     {fn:>6}       {tp:>6}")

    # Calculate precision and recall
    # Precision: Of all instances predicted as positive, how many were actually positive?
    # Recall: Of all actual positive instances, how many were correctly predicted?
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    print(f"\nThreshold: {threshold:.2f}")
    print(f"Precision ({class_names[1]}): {precision:.4f}")
    print(f"Recall    ({class_names[1]}): {recall:.4f}")


def main():
    # Load the raw test dataset
    _, _, test_data_raw, data_info = load_raw_data()
    class_names = data_info.features["label"].names

    # Prepare a batched test dataset for evaluation
    test_data = prepare_dataset(test_data_raw, batch_size=BATCH_SIZE, shuffle=False, do_batch=True)

    # Load the trained model
    model = tf.keras.models.load_model(MODEL_PATH)

    # Evaluate the model
    evaluate(model, test_data, class_names, threshold=THRESHOLD)


if __name__ == "__main__":
    main()

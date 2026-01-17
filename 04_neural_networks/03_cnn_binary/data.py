import tensorflow as tf
import tensorflow_datasets as tfds

IMAGE_SIZE = (160, 160)  # (height, width)
SHUFFLE_BUFFER_SIZE = 1000
PIXEL_SCALE = 255.0
AUTOTUNE = tf.data.AUTOTUNE
SEED = 42


def load_raw_data(data_dir="./tfds_data"):
    """
    Loads the raw Cats vs Dogs dataset from TensorFlow Datasets splitting it into training, validation, and test sets.

    The data is labeled as follows:
        0 -> Cat
        1 -> Dog

    Parameters:
        - data_dir (str | Path):  Directory to store/load the dataset.

    Returns:
        - ds_train (tf.data.Dataset):       Training dataset (80% of data)
        - ds_val (tf.data.Dataset):         Validation dataset (10% of data)
        - ds_test (tf.data.Dataset):        Test dataset (10% of data)
        - ds_info (tfds.core.DatasetInfo):  Metadata about the dataset
    """
    (ds_train, ds_val, ds_test), ds_info = tfds.load(
        "cats_vs_dogs",
        data_dir=data_dir,
        split=["train[:80%]", "train[80%:90%]", "train[90%:]"],
        as_supervised=True,
        with_info=True
    )

    return ds_train, ds_val, ds_test, ds_info


def preprocess_image(image, label):
    """
    Resizes image to a fixed size, converts it to float32, and normalizes pixel values to the range [0, 1].

    Parameters:
        - image (tf.Tensor): Input image tensor.
        - label (tf.Tensor): Corresponding label tensor.

    Returns:
        - image (tf.Tensor): Resized and normalized image tensor.
        - label (tf.Tensor): Unchanged label.
    """
    image = tf.image.resize_with_pad(image, IMAGE_SIZE[0], IMAGE_SIZE[1])
    image = tf.cast(image, tf.float32) / PIXEL_SCALE
    return image, label


def prepare_dataset(data_set, batch_size=32, shuffle=False, do_batch=True):
    """
    Prepares the dataset by applying preprocessing, batching, shuffling (if specified), and prefetching.

    Parameters:
        - data_set (tf.data.Dataset): Input dataset.
        - batch_size (int): Size of the batches.
        - shuffle (bool): Whether to shuffle the dataset.

    Returns:
        - data_set (tf.data.Dataset): Prepared dataset.
    """
    data_set = data_set.map(preprocess_image, num_parallel_calls=AUTOTUNE)

    if shuffle:
        data_set = data_set.shuffle(SHUFFLE_BUFFER_SIZE,  seed=SEED)

    if do_batch:
        data_set = data_set.batch(batch_size)

    return data_set.prefetch(AUTOTUNE)

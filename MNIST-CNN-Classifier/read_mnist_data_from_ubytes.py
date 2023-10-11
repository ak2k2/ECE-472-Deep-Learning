import os
import struct

import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf


def read_idx_labels(filename):
    with open(filename, "rb") as f:
        # Read magic number and number of items
        magic, num = struct.unpack(">II", f.read(8))

        # Read labels
        labels = np.fromfile(f, dtype=np.uint8)

    return labels


def read_idx_images(filename):
    with open(filename, "rb") as f:
        # Read magic number, number of images, rows, and columns
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))

        # Read image data
        images = np.fromfile(f, dtype=np.uint8).reshape(num, rows, cols)

    return images


def load_mnist_data_from_ubytes(path):
    # Paths to the MNIST dataset files
    if path is None:
        path = "mnist_data"
    if path[-1] == "/":
        path = path[:-1]

    train_labels_path = path + "/train-labels-idx1-ubyte"
    train_images_path = path + "/train-images-idx3-ubyte"
    test_labels_path = path + "/t10k-labels-idx1-ubyte"
    test_images_path = path + "/t10k-images-idx3-ubyte"

    # Read the data
    train_labels = read_idx_labels(train_labels_path)
    train_images = read_idx_images(train_images_path)
    test_labels = read_idx_labels(test_labels_path)
    test_images = read_idx_images(test_images_path)

    return train_labels, train_images, test_labels, test_images


def get_shuffled_mnist_data(seed: int = 0) -> tuple:
    # Load the MNIST data
    train_labels, train_images, test_labels, test_images = load_mnist_data_from_ubytes(
        "mnist_data"
    )

    # # Random Shuffling
    # np.random.seed(seed)
    # shuffle_indices = np.random.permutation(len(train_labels))
    # train_labels = train_labels[shuffle_indices]
    # train_images = train_images[shuffle_indices]

    # Validation Split - First 10% of the training data
    validation_split = 0.1
    total_train_samples = len(train_labels)
    num_validation_samples = int(validation_split * total_train_samples)

    val_labels = train_labels[:num_validation_samples]
    val_images = train_images[:num_validation_samples]

    train_labels = train_labels[num_validation_samples:]
    train_images = train_images[num_validation_samples:]

    # Step 3: Normalization (0-255 to 0-1)
    train_images = train_images.astype("float32") / 255.0
    val_images = val_images.astype("float32") / 255.0
    test_images = test_images.astype("float32") / 255.0

    # Reshape the image data to include the channel dimension
    train_images = train_images.reshape((-1, 28, 28, 1))
    val_images = val_images.reshape((-1, 28, 28, 1))
    test_images = test_images.reshape((-1, 28, 28, 1))

    train_labels = tf.convert_to_tensor(train_labels, dtype=tf.int64)
    train_images = tf.convert_to_tensor(train_images, dtype=tf.float32)
    val_labels = tf.convert_to_tensor(val_labels, dtype=tf.int64)
    val_images = tf.convert_to_tensor(val_images, dtype=tf.float32)
    test_labels = tf.convert_to_tensor(test_labels, dtype=tf.int64)
    test_images = tf.convert_to_tensor(test_images, dtype=tf.float32)

    return train_labels, train_images, val_labels, val_images, test_labels, test_images

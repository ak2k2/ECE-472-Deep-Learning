import os
import pickle

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# SOURCE: https://www.cs.toronto.edu/~kriz/cifar.html
# Citation: Learning Multiple Layers of Features from Tiny Images, Alex Krizhevsky, 2009
# https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf


def unpickle(file):
    """
    Loaded in this way, each of the batch files contains a dictionary with the following elements:
    data -- a 10000x3072 numpy array of uint8s. Each row of the array stores a 32x32 colour image. The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue.
    The image is stored in row-major order, so that the first 32 entries of the array are the red channel values of the first row of the image.
    labels -- a list of 10000 numbers in the range 0-9. The number at index i indicates the label of the ith image in the array data.

    The dataset contains another file, called batches.meta. It too contains a Python dictionary object. It has the following entries:
    label_names -- a 10-element list which gives meaningful names to the numeric labels in the labels array described above. For example, label_names[0] == "airplane", label_names[1] == "automobile", etc.
    """

    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict


def get_cifar_10(parent="./cifar_pickle_archive/cifar-10-batches-py") -> dict:
    """
    Input: parent: str
        Path to the cifar-10-batches-py directory
    Output: dict
        A dictionary containing the cifar-10 data. The keys are:

        meta: dict
            A dictionary mapping classification classes to human readable label names.
        train: list
            A list of dictionaries, each containing the data and labels for a given batch of training data.
        test: dict
            A dictionary containing the data and labels for the test data.
    """
    batches = [
        "data_batch_1",
        "data_batch_2",
        "data_batch_3",
        "data_batch_4",
        "data_batch_5",
        "test_batch",
    ]
    meta = "batches.meta"

    data = {}
    data["meta"] = unpickle(f"{parent}/{meta}")
    data["train"] = []
    for batch in batches:
        data["train"].append(unpickle(f"{parent}/{batch}"))

    data["test"] = unpickle(f"{parent}/test_batch")

    return data


def create_dataframe_with_labels(cifar_data, meta_data, batch_type="train"):
    images = []
    labels = []
    label_names = []

    # Extract label names from meta data
    label_names_dict = {
        i: name.decode("utf-8") for i, name in enumerate(meta_data[b"label_names"])
    }

    # Function to process a single batch of data
    def process_batch(batch):
        for i in range(len(batch[b"data"])):
            image_flat = batch[b"data"][i]
            image_red = image_flat[:1024].reshape(32, 32)
            image_green = image_flat[1024:2048].reshape(32, 32)
            image_blue = image_flat[2048:].reshape(32, 32)
            image_reshaped = np.dstack((image_red, image_green, image_blue))
            images.append(image_reshaped)

            label = batch[b"labels"][i]
            labels.append(label)
            label_names.append(label_names_dict[label])

    # Process data based on batch type
    if batch_type == "train":
        for batch in cifar_data[batch_type]:
            process_batch(batch)
    else:  # For test data
        process_batch(cifar_data[batch_type])

    return pd.DataFrame({"image": images, "label": labels, "label_name": label_names})


def display_image_from_df(df, index):
    """
    Display an image from the DataFrame.

    Args:
    df (pd.DataFrame): DataFrame containing the CIFAR-10 data.
    index (int): Index of the image to be displayed.
    """
    # Extract and display the image data directly from the DataFrame
    image_data = df.iloc[index]["image"]

    # Display the image
    plt.imshow(image_data, interpolation="nearest")
    plt.title(f"Label: {df.iloc[index]['label_name']}")
    plt.axis("off")  # Turn off axis numbers
    plt.show()


def get_cifar_10_as_df():
    cifar_10 = get_cifar_10()
    train = create_dataframe_with_labels(cifar_10, cifar_10["meta"], batch_type="train")
    test = create_dataframe_with_labels(cifar_10, cifar_10["meta"], batch_type="test")
    return train, test

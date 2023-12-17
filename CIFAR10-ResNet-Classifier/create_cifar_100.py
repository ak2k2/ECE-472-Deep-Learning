import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def unpickle_cifar100(file):
    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict


def create_cifar100_df(data_batch, meta_data):
    images = []
    fine_labels = []
    coarse_labels = []
    fine_label_names = []
    coarse_label_names = []

    fine_label_names_dict = {
        i: name.decode("utf-8") for i, name in enumerate(meta_data[b"fine_label_names"])
    }
    coarse_label_names_dict = {
        i: name.decode("utf-8")
        for i, name in enumerate(meta_data[b"coarse_label_names"])
    }

    for i in range(len(data_batch[b"data"])):
        image_flat = data_batch[b"data"][i]
        image_red = image_flat[:1024].reshape(32, 32)
        image_green = image_flat[1024:2048].reshape(32, 32)
        image_blue = image_flat[2048:].reshape(32, 32)
        image_reshaped = np.dstack((image_red, image_green, image_blue))
        images.append(image_reshaped)

        fine_label = data_batch[b"fine_labels"][i]
        coarse_label = data_batch[b"coarse_labels"][i]
        fine_labels.append(fine_label)
        coarse_labels.append(coarse_label)
        fine_label_names.append(fine_label_names_dict[fine_label])
        coarse_label_names.append(coarse_label_names_dict[coarse_label])

    return pd.DataFrame(
        {
            "image": images,
            "fine_label": fine_labels,
            "coarse_label": coarse_labels,
            "fine_label_name": fine_label_names,
            "coarse_label_name": coarse_label_names,
        }
    )


def get_cifar_100_as_df(parent="./cifar_pickle_archive/cifar-100-python"):
    meta = unpickle_cifar100(f"{parent}/meta")
    train_data = unpickle_cifar100(f"{parent}/train")
    test_data = unpickle_cifar100(f"{parent}/test")

    train_df = create_cifar100_df(train_data, meta)
    test_df = create_cifar100_df(test_data, meta)

    return train_df, test_df


def display_image_from_df_cifar100(df, index):
    image_data = df.iloc[index]["image"]
    plt.imshow(image_data, interpolation="nearest")
    plt.title(
        f"Fine Label: {df.iloc[index]['fine_label_name']}, Coarse Label: {df.iloc[index]['coarse_label_name']}"
    )
    plt.axis("off")
    plt.show()

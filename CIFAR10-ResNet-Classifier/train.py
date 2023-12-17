import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import yaml
from tqdm import trange

from adam_l2 import AdamWithL2Regularization
from create_cifar_10 import get_cifar_10_as_df
from create_cifar_100 import get_cifar_100_as_df
from res_block_net import Classifier

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def preprocess_data(df, label_column):
    images = np.stack(df["image"].values) / 255.0
    labels = tf.keras.utils.to_categorical(
        df[label_column], num_classes=df[label_column].nunique()
    )
    return images, labels


def initialize_model(num_classes):
    return Classifier(num_classes=num_classes)


def train_model(
    model,
    train_images,
    train_labels,
    test_images,
    test_labels,
    config,
):
    num_epochs = config["num_epochs"]
    batch_size = config["batch_size"]

    optimizer = AdamWithL2Regularization(learning_rate=0.005, lambda_l2=0.0001)
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    for epoch in range(num_epochs):
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.CategoricalAccuracy()

        for i in trange(0, len(train_images), batch_size):
            batch_x = train_images[i : i + batch_size]
            batch_y = train_labels[i : i + batch_size]

            with tf.GradientTape() as tape:
                logits = model(batch_x)
                loss_value = loss_fn(batch_y, logits)

            gradients = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(gradients, model.trainable_variables)

            epoch_loss_avg.update_state(loss_value)
            epoch_accuracy.update_state(batch_y, logits)

        train_loss = epoch_loss_avg.result()
        train_accuracy = epoch_accuracy.result()
        test_loss, test_accuracy = evaluate_model(model, test_images, test_labels)

        print(
            f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.3f}, Accuracy: {train_accuracy:.3%}"
        )
        print(f"Test Loss: {test_loss:.3f}, Test Accuracy: {test_accuracy:.3%}")


def evaluate_model(model, test_images, test_labels):
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    test_logits = model(test_images)
    test_loss = loss_fn(test_labels, test_logits)
    test_accuracy = tf.keras.metrics.categorical_accuracy(test_labels, test_logits)
    return test_loss.numpy(), np.mean(test_accuracy)


def display_sample_predictions(
    model, test_images, test_labels, num_samples=10, dataset_name="CIFAR-10"
):
    indices = np.random.choice(len(test_images), num_samples, replace=False)
    sample_images = test_images[indices]
    sample_labels = test_labels[indices]

    predictions = model(sample_images)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(sample_labels, axis=1)

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        img = sample_images[i]
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(
            f"{dataset_name} - True: {true_classes[i]}, Predicted: {predicted_classes[i]}"
        )

    plt.tight_layout()
    plt.show()


def run_pipeline(config):
    dataset_type = config["dataset_type"]
    num_training_examples = config["num_training_examples"]

    if dataset_type == 10:
        train_df, test_df = get_cifar_10_as_df()
        label_column = "label"
    elif dataset_type == 100:
        train_df, test_df = get_cifar_100_as_df()
        label_column = "fine_label"
    else:
        raise ValueError("Invalid dataset type. Choose 10 or 100.")

    num_training_examples = 1000
    train_df = train_df[:num_training_examples]
    test_df = test_df[:num_training_examples]

    train_images, train_labels = preprocess_data(train_df, label_column)
    test_images, test_labels = preprocess_data(test_df, label_column)

    model = initialize_model(num_classes=train_df[label_column].nunique())
    train_model(model, train_images, train_labels, test_images, test_labels, config)
    display_sample_predictions(
        model, test_images, test_labels, dataset_type, f"CIFAR-{dataset_type}"
    )


if __name__ == "__main__":
    config = load_config()
    # run_pipeline(10)  # For CIFAR-10
    run_pipeline(config)

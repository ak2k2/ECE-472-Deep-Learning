import os

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

from custom_adam import AdamWithL2Regularization
from create_cifar_df import get_cifar_10_as_df

from classes import Conv2d, GroupNorm, ResidualBlock, Classifier


def load_config():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config


def cross_entropy_loss(y_pred, y):
    sparse_ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=y_pred)
    return tf.reduce_mean(sparse_ce)


def accuracy(y_pred, y):
    class_preds = tf.argmax(tf.nn.softmax(y_pred), axis=1)
    is_equal = tf.equal(y, class_preds)
    return tf.reduce_mean(tf.cast(is_equal, tf.float32))


def train(
    train_images,
    train_labels,
    test_images,
    test_labels,
    model,
    optimizer,
    loss_fn,
    train_acc_metric,
    val_acc_metric,
    num_epochs=10,
    batch_size=64,
):
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            logits = model.forward(images)
            loss_value = loss_fn(labels, logits)
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        train_acc_metric.update_state(labels, logits)
        return loss_value

    def test_step(images, labels):
        logits = model.forward(images)
        val_acc_metric.update_state(labels, logits)
        # Training loop
        batch_size = 64
        num_epochs = 10

    for epoch in range(num_epochs):
        print("\nStart of epoch %d" % (epoch,))

        # Iterate over the batches of the dataset
        for step in range(0, len(train_images), batch_size):
            x_batch_train = train_images[step : step + batch_size]
            y_batch_train = train_labels[step : step + batch_size]

            loss_value = train_step(x_batch_train, y_batch_train)

            # Log every 200 batches
            if step % 200 == 0:
                print(f"Training loss (for one batch) at step {step}: {loss_value}")
                print(f"Seen so far: {(step + 1) * batch_size} samples")

        # Display metrics at the end of each epoch
        train_acc = train_acc_metric.result()
        print(f"Training accuracy over epoch: {train_acc}")
        train_acc_metric.reset_states()

        # Run a validation loop at the end of each epoch
        for step in range(0, len(test_images), batch_size):
            x_batch_val = test_images[step : step + batch_size]
            y_batch_val = test_labels[step : step + batch_size]
            test_step(x_batch_val, y_batch_val)

        val_acc = val_acc_metric.result()
        val_acc_metric.reset_states()
        print(f"Validation accuracy: {val_acc}")


train_df, test_df = get_cifar_10_as_df()

train_images = np.array(train_df["image"].tolist())
train_labels = np.array(train_df["label"].tolist())
test_images = np.array(test_df["image"].tolist())
test_labels = np.array(test_df["label"].tolist())

model = Classifier(num_classes=10)
optimizer = AdamWithL2Regularization()
loss_fn = cross_entropy_loss
train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

train(
    train_images,
    train_labels,
    test_images,
    test_labels,
    model,
    optimizer,
    loss_fn,
    train_acc_metric,
    val_acc_metric,
    num_epochs=10,
    batch_size=64,
)

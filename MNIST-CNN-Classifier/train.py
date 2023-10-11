import os
import yaml

import matplotlib.pyplot as plt
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

from classifier import Classifier
from read_mnist_data_from_ubytes import get_shuffled_mnist_data
from custom_adam import AdamWithL2Regularization


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


def get_loss_and_accuracy_from_dataset(model, images, labels):
    y_pred = model.forward(images)
    get_loss = cross_entropy_loss(y_pred, labels)
    get_accuracy = accuracy(y_pred, labels)
    return get_loss.numpy(), get_accuracy.numpy()


def main():
    config = load_config()

    tf.random.set_seed(config["tf_random_seed"])
    np.random.seed(config["np_random_seed"])

    # Load MNIST data from ubyte wrapper
    (
        train_labels,
        train_images,
        val_labels,
        val_images,
        test_labels,
        test_images,
    ) = get_shuffled_mnist_data(seed=0)

    NUM_TRAINING_SAMPLES = config["num_training_samples"]
    train_labels = train_labels[:NUM_TRAINING_SAMPLES]
    train_images = train_images[:NUM_TRAINING_SAMPLES]

    classifier = Classifier(
        input_depth=config["input_depth"],
        conv_layer_depths=config["conv_layer_depths"],
        conv_kernel_sizes=config["conv_kernel_sizes"],
        fc_layer_sizes=config["fc_layer_sizes"],
        num_classes=config["num_classes"],
        dropout_rate=config["dropout_rate"],
    )

    optimizer = AdamWithL2Regularization(
        learning_rate=config["learning_rate"],
        beta_1=config["beta_1"],
        beta_2=config["beta_2"],
        lambda_l2=config["lambda_l2"],
    )

    epochs = config["epochs"]
    batch_size = config["batch_size"]

    # Shuffle and batch training and validation sets
    train_dataset = (
        tf.data.Dataset.from_tensor_slices((train_images, train_labels))
        .shuffle(buffer_size=len(train_images) + 1)
        .batch(batch_size)
    )

    validation_dataset = (
        tf.data.Dataset.from_tensor_slices((val_images, val_labels))
        .shuffle(buffer_size=len(val_images) + 1)
        .batch(batch_size)
    )

    global_train_loss = []
    global_train_accuracy = []
    global_val_loss = []
    global_val_accuracy = []

    print(
        "num_params",
        tf.math.add_n(
            [tf.math.reduce_prod(var.shape) for var in classifier.trainable_variables]
        ),
    )

    for epoch in range(epochs):
        # Initialize metrics for the epoch
        epoch_loss_sum = 0.0
        epoch_accuracy_sum = 0.0
        batch_count = 0

        for x_batch, y_batch in train_dataset:
            with tf.GradientTape() as tape:
                y_pred = classifier.forward(x_batch)
                batch_loss = cross_entropy_loss(y_pred, y_batch)

            grads = tape.gradient(batch_loss, classifier.trainable_variables)
            optimizer.apply_gradients(grads, classifier.trainable_variables)

            # Update loss and accuracy for the epoch
            epoch_loss_sum += batch_loss.numpy()
            epoch_accuracy_sum += accuracy(y_pred, y_batch).numpy()
            batch_count += 1

        # Calculate average training loss and accuracy for the epoch
        val_loss_sum = 0.0
        val_accuracy_sum = 0.0
        val_batch_count = 0

        for x_val, y_val in validation_dataset:
            val_loss, val_accuracy = get_loss_and_accuracy_from_dataset(
                classifier, x_val, y_val
            )
            val_loss_sum += val_loss
            val_accuracy_sum += val_accuracy
            val_batch_count += 1

        avg_val_loss = val_loss_sum / val_batch_count
        avg_val_accuracy = val_accuracy_sum / val_batch_count

        global_val_loss.append(avg_val_loss)
        global_val_accuracy.append(avg_val_accuracy)

        global_train_loss.append(epoch_loss_sum / batch_count)
        global_train_accuracy.append(epoch_accuracy_sum / batch_count)

        avg_train_loss = epoch_loss_sum / batch_count
        avg_train_accuracy = epoch_accuracy_sum / batch_count

        print(
            f"Epoch {epoch+1}: "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Train Accuracy: {avg_train_accuracy * 100:.4f}%, "
            f"Validation Loss: {avg_val_loss:.4f}, "
            f"Validation Accuracy: {avg_val_accuracy * 100:.4f}%"
        )

    (  # Evaluate on the entire test set
        test_loss,
        test_accuracy,
    ) = get_loss_and_accuracy_from_dataset(classifier, test_images, test_labels)

    print(f"Loss on the test set: {test_loss:.4f}")
    print(f"Accuracy on the test set: {test_accuracy * 100:.2f}%")

    plt.plot(global_train_loss, label="Training Loss")
    plt.plot(global_val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    plt.plot(global_train_accuracy, label="Training Accuracy")
    plt.plot(global_val_accuracy, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    # Select 10 random images from the test set
    num_images = 10
    random_indices = np.random.choice(
        test_images.shape[0], size=num_images, replace=False
    )
    random_test_images = test_images.numpy()[
        random_indices
    ]  # Convert to NumPy array for advanced indexing∏
    random_test_labels = test_labels.numpy()[random_indices]

    # Get predictions
    predictions = classifier.forward(random_test_images)
    predicted_labels = tf.argmax(tf.nn.softmax(predictions), axis=1)

    # Plot images with true and predicted labels
    plt.figure(figsize=(15, 5))
    for i in range(num_images):
        plt.subplot(2, 5, i + 1)
        plt.imshow(random_test_images[i].reshape(28, 28), cmap="gray")
        plt.title(f"True: {random_test_labels[i]}, Pred: {predicted_labels[i].numpy()}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

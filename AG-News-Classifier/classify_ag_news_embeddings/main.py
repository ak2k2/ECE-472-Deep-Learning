import sys
from pathlib import Path
import yaml

import numpy as np
import pandas as pd
import tensorflow as tf
import tqdm
from custom_adam import AdamWithL2Regularization

# Assuming the other modules (Linear, MLP, AdamWithL2Regularization) are in the same directory
from linear_module import Linear
from mlp_module import MLP
from sklearn.metrics import confusion_matrix

script_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(script_dir))

from load_ag_news_dataset.read_ag_news_data import (  # reuse the load_data function from read_ag_news_data.py
    load_data,
)


def load_config(config_path):
    with open(config_path, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            sys.exit(1)  # Exit the script with an error code


def get_embedded_dfs(config):
    test_df = load_data(config["dataset"]["test_pickle_path"])
    train_df = load_data(config["dataset"]["train_pickle_path"])
    return test_df, train_df


def train_model(
    model,
    optimizer,
    features,
    labels,
    num_epochs,
    batch_size,
):
    for epoch in range(num_epochs):
        # Shuffle data
        indices = np.arange(len(features))
        np.random.shuffle(indices)
        features = features[indices]
        labels = labels[indices]

        for i in tqdm.trange(0, len(features), batch_size):
            # Create batches
            feature_batch = features[i : i + batch_size]
            label_batch = labels[i : i + batch_size]

            # Forward pass
            with tf.GradientTape() as tape:
                logits = model(feature_batch)
                loss = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=logits, labels=label_batch
                    )
                )

            # Backward pass
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(grads, model.trainable_variables)

        print(f"Epoch {epoch+1}, Loss: {loss.numpy()}")
        # print test accuracy
        logits = model(features)
        predicted_classes = tf.argmax(logits, axis=1)
        accuracy = np.mean(predicted_classes.numpy() == labels)
        print(f"Train accuracy: {accuracy*100:.2f}%")


# Evaluation on the test set
def evaluate_model(model, features, labels):
    logits = model(features)
    predicted_classes = tf.argmax(logits, axis=1)
    accuracy = np.mean(predicted_classes.numpy() == labels)
    print(f"Test accuracy: {accuracy*100:.2f}%")
    return predicted_classes.numpy()  # Return the predicted classes


def print_padding():
    print("\n\n")
    print("*" * 80)
    print("*" * 80)


def main():
    config_path = (
        "classify_ag_news_embeddings/config.yaml"  # Path to the configuration file
    )
    config = load_config(config_path)  # Load the configuration

    np.random.seed(config["seed"])
    tf.random.set_seed(config["seed"])

    # Prepare data
    test_df, train_df = get_embedded_dfs(config)
    train_labels = train_df["label"].values
    train_features = np.stack(train_df["embedding"].values).astype(
        np.float32
    )  # Cast to float32
    test_labels = test_df["label"].values
    test_features = np.stack(test_df["embedding"].values).astype(
        np.float32
    )  # Cast to float32

    # Model Parameters
    num_inputs = len(
        train_features[0]
    )  # assuming the length of embedding vectors is consistent
    num_outputs = 4  # 4 classes in the AG News dataset

    # Load hyperparameters from the config file
    num_hidden_layers = config["model"]["num_hidden_layers"]
    hidden_layer_width = config["model"]["hidden_layer_width"]
    hidden_activation = (
        tf.nn.relu6 if config["model"]["hidden_activation"] == "relu" else tf.identity
    )
    output_activation = tf.identity

    # Create the MLP model
    model = MLP(
        num_inputs,
        num_outputs,
        num_hidden_layers,
        hidden_layer_width,
        hidden_activation,
        output_activation,
        seed=config["seed"],
    )

    # Custom Adam with L2 regularization
    optimizer = AdamWithL2Regularization(
        learning_rate=config["optimizer"]["learning_rate"],
        beta_1=config["optimizer"]["beta_1"],
        beta_2=config["optimizer"]["beta_2"],
        ep=config["optimizer"]["ep"],
        lambda_l2=config["optimizer"]["lambda_l2"],
    )

    print_padding()
    print("Starting training ...\n")
    print(f"Model: {model}\n")
    print(f"Optimizer: {optimizer}\n")
    print(f"Train features shape: {train_features.shape}")
    print(f"Train labels shape: {train_labels.shape}\n")
    print(f"Test features shape: {test_features.shape}")
    print(f"Test labels shape: {test_labels.shape}\n")
    print(f"Number of trainable variables: {len(model.trainable_variables)}\n")
    print_padding()

    train_model(
        model,
        optimizer,
        train_features,
        train_labels,
        num_epochs=config["training"]["num_epochs"],
        batch_size=config["training"]["batch_size"],
    )

    # Evaluate
    predicted_classes = evaluate_model(model, test_features, test_labels)

    # Display a sample of test set items with their true and predicted labels
    num_display_items = config["evaluation"]["num_display_items"]
    sample_indices = np.random.choice(
        len(test_labels), num_display_items, replace=False
    )
    sample_texts = test_df.iloc[sample_indices]["text"]
    sample_true_labels = test_labels[sample_indices]
    sample_predicted_labels = predicted_classes[sample_indices]

    display_df = pd.DataFrame(
        {
            "Text": sample_texts,
            "True Label": sample_true_labels,
            "Predicted Label": sample_predicted_labels,
        }
    )
    print(display_df)

    # Confusion matrix
    matrix = confusion_matrix(test_labels, predicted_classes)

    print_padding()
    print("Confusion Matrix:")
    print(matrix)
    print_padding()


if __name__ == "__main__":
    main()

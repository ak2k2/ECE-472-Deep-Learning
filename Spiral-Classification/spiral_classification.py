import logging
import os

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import yaml
from tqdm import trange

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # disable tensorflow warnings
import tensorflow as tf
from sklearn.inspection import DecisionBoundaryDisplay

from generate_spirals import generate_spiral_data
from mlp_module import MLP


def setup_logging(level) -> logging.Logger:
    """Setup logging configuration"""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def load_config(path="config.yaml"):
    """Load config from yaml file"""
    with open(path, "r") as f:
        return yaml.safe_load(f)  # Use PyYAML's safe_load method to read the YAML file


def log_cross_entropy_loss(y_true, y_pred):
    """Compute mean binary cross-entropy loss, mimicking tf.keras.losses.BinaryCrossentropy"""
    epsilon = 1e-7  # To prevent log(0)

    # Clip predictions to avoid log(0)
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

    # Compute binary cross-entropy
    loss = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)

    # Compute the mean over the batch
    return tf.reduce_mean(loss)


# Function to compute L2 regularization loss
def l2_regularization(layers, lambda_reg):
    reg_loss = 0.0
    for layer in layers:
        reg_loss += tf.reduce_sum(tf.square(layer.w))
    return lambda_reg * reg_loss


def main():
    # Load config
    logger = setup_logging(logging.INFO)
    config = load_config()

    (
        seed,
        N,
        K,
        num_inputs,
        num_outputs,
        num_hidden_layers,
        hidden_layer_width,
        learning_rate,
        epochs,
        lambda_reg,
    ) = (
        config["seed"],
        config["N"],
        config["K"],
        config["num_inputs"],
        config["num_outputs"],
        config["num_hidden_layers"],
        config["hidden_layer_width"],
        config["learning_rate"],
        config["epochs"],
        config["lambda_reg"],
    )

    logger.info(f"Config: {config}")

    # Set rng seeds
    tf.random.set_seed(seed)
    np.random.seed(seed)

    # Generate spiral data
    x_data, y_data = generate_spiral_data(seed=seed, N=N, K=K)
    x_data = tf.constant(x_data, dtype=tf.float32)
    y_data = tf.constant(y_data, dtype=tf.float32)

    activation_mapping = {
        "relu": tf.nn.relu,
        "leaky_relu": tf.nn.leaky_relu,
        "sigmoid": tf.sigmoid,
        "tanh": tf.tanh,
    }

    hidden_activation = activation_mapping.get(config["hidden_activation"], tf.identity)
    output_activation = activation_mapping.get(config["output_activation"], tf.identity)

    # Initialize MLP model using parameters from config
    mlp = MLP(
        num_inputs,
        num_outputs,
        num_hidden_layers,
        hidden_layer_width,
        hidden_activation,
        output_activation,
        seed,
    )

    optimizer = tf.optimizers.Adam(learning_rate, beta_1=0.99, beta_2=0.999)

    fig, ax = plt.subplots()

    # Create mesh grid outside the loop
    x_min, x_max = x_data[:, 0].numpy().min() - 1, x_data[:, 0].numpy().max() + 1
    y_min, y_max = x_data[:, 1].numpy().min() - 1, x_data[:, 1].numpy().max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

    def update(epoch):
        # Training and prediction logic here
        with tf.GradientTape() as tape:
            y_pred = mlp(x_data)
            loss = log_cross_entropy_loss(y_data, y_pred)
            loss += l2_regularization(mlp.layers, lambda_reg)

        grads = tape.gradient(loss, mlp.trainable_variables)
        optimizer.apply_gradients(zip(grads, mlp.trainable_variables))

        if epoch % 10 == 0:
            ax.clear()  # Clear the plot only when redrawing
            print(f"Epoch {epoch}, Loss: {loss}")

            # Predict labels for each point in mesh grid
            Z = mlp(tf.constant(np.c_[xx.ravel(), yy.ravel()], dtype=tf.float32))
            Z = Z.numpy().reshape(xx.shape)

            # Use sklearn's DecisionBoundaryDisplay
            display = DecisionBoundaryDisplay(xx0=xx, xx1=yy, response=Z)
            display.plot(ax=ax)
            ax.scatter(
                x_data[:, 0],
                x_data[:, 1],
                c=y_data[:, 0],
                edgecolors="k",
                marker="o",
                s=25,
                linewidth=1,
            )
            ax.set_title("Decision Boundary")

    ani = animation.FuncAnimation(fig, update, frames=range(epochs), repeat=False)

    # Save the animation as an MP4 video
    ani.save("decision_boundary_animation_2.mp4", writer="ffmpeg", fps=60)


if __name__ == "__main__":
    main()

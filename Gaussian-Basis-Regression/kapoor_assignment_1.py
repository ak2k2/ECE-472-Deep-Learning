import os
import json
import logging

import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # disable tensorflow warnings
import tensorflow as tf


def setup_logging(level):
    """Setup logging configuration"""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def load_config(path="config.json"):
    """Load config from json file"""
    with open(path, "r") as f:
        return json.load(f)


def plot_error(x_values, y_true, y_pred, title):
    error = np.abs(y_true - y_pred)
    plt.figure()
    plt.title(title)
    plt.plot(x_values, error, label="Realized Error over a Denser Range")
    plt.xlabel("x")
    plt.ylabel("|y - ŷ|")
    plt.legend()
    plt.show()


class Linear(tf.Module):
    def __init__(self, num_basis, seed):
        initializer = tf.initializers.GlorotNormal(seed)
        self.w = tf.Variable(initializer([num_basis, 1]), trainable=True)
        self.b = tf.Variable([0.0], trainable=True)

    def __call__(self, x):
        return tf.matmul(x, self.w) + self.b  # y = w^T * x + b


# Basis Expansion: Gaussian Basis Functions
class BasisExpansion(tf.Module):
    def __init__(self, num_basis, RANGE_END):
        self.mu = tf.Variable(
            tf.linspace(-1, 1, num_basis)[tf.newaxis, :],
            trainable=True,
        )
        self.sigma = tf.Variable(tf.ones([1, num_basis]), trainable=True)

    def __call__(self, x):
        return tf.exp(
            -tf.square((x - self.mu) / self.sigma)
        )  # BASIS FUNCTIONS: phi = exp(-(x-mu)^2/sigma^2)


def main(
    SEED,
    RANGE_END,
    NUM_DATA_POINTS,
    NUM_BASIS_FUNCTIONS,
    STEP_SIZE,
    NUM_ITERS,
    LOG_LEVEL,
):
    def target(x):
        return tf.math.sin(x)

    logger = setup_logging(LOG_LEVEL)
    rng = np.random.default_rng(seed=SEED)
    plt.rcParams["figure.figsize"] = [18, 10]
    tf.config.threading.set_intra_op_parallelism_threads(os.cpu_count())
    tf.random.set_seed(SEED)

    # x = tf.linspace(-RANGE_END * np.pi / 2, RANGE_END * np.pi / 2, NUM_DATA_POINTS)[
    #     :, tf.newaxis
    # ]

    x = tf.random.uniform((NUM_DATA_POINTS, 1), 0, 1)
    y = target(x) + rng.normal(0, 0.1, (NUM_DATA_POINTS, 1))

    linear = Linear(NUM_BASIS_FUNCTIONS, SEED)
    basis_expansion = BasisExpansion(NUM_BASIS_FUNCTIONS, RANGE_END)

    optimizer = tf.optimizers.SGD(STEP_SIZE)

    losses = {}
    plt.ion()  # Turn on interactive mode

    for i in trange(NUM_ITERS):
        with tf.GradientTape() as tape:
            phi = basis_expansion(x)
            y_hat = linear(phi)
            loss = tf.reduce_mean(0.5 * tf.square(y - y_hat))
            losses[i] = loss.numpy()

        grads = tape.gradient(
            loss, [linear.w, linear.b, basis_expansion.mu, basis_expansion.sigma]
        )
        optimizer.apply_gradients(
            zip(grads, [linear.w, linear.b, basis_expansion.mu, basis_expansion.sigma])
        )
        optimizer.learning_rate.assign(optimizer.learning_rate * 0.999)

        if i % 20 == 0:  # Refresh Frame every 10 iterations
            plt.clf()
            plt.title(
                f"Using SGD to Optimize Linear Model and Gaussian Basis Expansion Weights and Biases: Iteration={i}"
            )
            plt.xlabel("x")
            plt.ylabel("y")
            plt.scatter(x, y, label=f"{NUM_DATA_POINTS} Noisy Samples", c="blue")
            plt.plot(x, target(x), label="Pure Sine", c="green")
            plt.plot(
                x,
                linear(basis_expansion(x)),
                label="Gaussian Regression Model",
                c="red",
                linestyle="dashed",
            )

            # Overlay individual Gaussian basis functions
            w = linear.w.numpy().flatten()
            for j, weight in enumerate(w):
                if weight >= 0:
                    plt.plot(
                        x, basis_expansion(x)[:, j], label=f"Basis {j+1}", alpha=0.3
                    )
                else:
                    plt.plot(
                        x,
                        -basis_expansion(x)[:, j],
                        label=f"Basis {j+1} (Flipped)",
                        alpha=0.3,
                    )

            # Generate concise equation text
            w = linear.w.numpy().flatten()
            b = linear.b.numpy()[0]
            mu_vals = basis_expansion.mu.numpy().flatten()
            sigma_vals = basis_expansion.sigma.numpy().flatten()

            equation_str = f"y = {b:.4f} "
            for j, (weight, mu_val, sigma_val) in enumerate(
                zip(w, mu_vals, sigma_vals)
            ):
                equation_str += f"+ {weight:.4f} * Φ({mu_val:.2f}, {sigma_val:.2f}) "

            # Display equation
            plt.text(
                0.05,
                0.95,
                equation_str,
                transform=plt.gca().transAxes,
                fontsize=9,
                verticalalignment="top",
            )

            plt.legend()
            plt.draw()
            plt.pause(0.1)

    plt.ioff()  # Turn off interactive mode

    # Final loss
    logger.info(f"Final loss: {losses[NUM_ITERS-1]}")

    # Final R^2
    y_hat = linear(basis_expansion(x))
    y_bar = tf.reduce_mean(y)
    ss_tot = tf.reduce_sum(tf.square(y - y_bar))
    ss_res = tf.reduce_sum(tf.square(y - y_hat))
    r_squared = 1 - ss_res / ss_tot

    logger.info(f"Final R^2: {r_squared}")

    # Indivisual Gaussian Basis Functions
    plt.figure()
    plt.title("Indivisual Gaussian Basis Functions")
    for j in range(NUM_BASIS_FUNCTIONS):
        plt.plot(x, basis_expansion(x)[:, j], label=f"Basis {j+1}")
    plt.legend()
    plt.show()

    # Loss vs Iteration
    plt.figure()
    plt.title("Loss vs Iteration")
    plt.plot(list(losses.keys()), list(losses.values()))
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.show()

    # Generate denser linspce between -2π and 2π
    new_x = tf.linspace(-RANGE_END * np.pi, RANGE_END * np.pi, 10000)[:, tf.newaxis]
    new_y_true = tf.math.sin(new_x)

    # Use the model to make predictions on the new data
    new_y_pred = linear(basis_expansion(new_x))

    # Plot the error
    plot_error(new_x, new_y_true, new_y_pred, "Error for values between -2π and 2π")


if __name__ == "__main__":
    config = load_config()
    main(**config)

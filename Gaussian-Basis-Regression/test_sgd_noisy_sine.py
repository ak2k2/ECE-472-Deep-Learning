import unittest
import json
import os
from parameterized import parameterized

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # disable tensorflow warnings


import tensorflow as tf

from kapoor_assignment_1 import (
    load_config,
    setup_logging,
    BasisExpansion,
    Linear,
)


class Test(unittest.TestCase):
    def setUp(self):
        self.config = {
            "SEED": 1401258965930102242,
            "RANGE_END": 4,
            "NUM_DATA_POINTS": 50,
            "NUM_BASIS_FUNCTIONS": 6,
            "STEP_SIZE": 0.1,
            "NUM_ITERS": 1000,
            "LOG_LEVEL": "INFO",
        }
        self.linear = Linear(self.config["NUM_BASIS_FUNCTIONS"], self.config["SEED"])
        self.basis_expansion = BasisExpansion(
            self.config["NUM_BASIS_FUNCTIONS"], self.config["RANGE_END"]
        )

    def test_load_config(self):
        # Assuming config.json has the expected content
        with open("config.json", "w") as f:
            json.dump(self.config, f)
        loaded_config = load_config("config.json")
        self.assertEqual(loaded_config, self.config)

    def test_setup_logging(self):
        logger = setup_logging("INFO")
        self.assertEqual(logger.level, 0)

    def test_basis_expansion(self):
        x = tf.constant([[0.0], [1.0], [2.0]])
        phi = self.basis_expansion(x)
        self.assertEqual(phi.shape, (3, self.config["NUM_BASIS_FUNCTIONS"]))

    def test_linear(self):
        # Prepare an input tensor x with shape [1, num_basis]
        num_basis = 6  # Adjust as needed
        x = tf.constant([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]], dtype=tf.float32)

        # Initialize Linear class
        seed = 0  # Your seed
        self.linear = Linear(num_basis, seed)

        # Call and check shape
        output = self.linear(x)
        self.assertEqual(output.shape, [1, 1])

    def test_loss_decrease(self):
        x = tf.constant([[0.0], [1.0], [2.0]], dtype=tf.float32)
        y = tf.constant([[0.0], [1.0], [2.0]], dtype=tf.float32)

        optimizer = tf.optimizers.SGD(0.1)

        initial_loss = tf.reduce_mean(
            0.5 * tf.square(y - self.linear(self.basis_expansion(x)))
        )

        with tf.GradientTape() as tape:
            tape.watch([self.linear.w, self.linear.b])
            y_hat = self.linear(self.basis_expansion(x))
            loss = tf.reduce_mean(0.5 * tf.square(y - y_hat))

        grads = tape.gradient(loss, [self.linear.w, self.linear.b])
        optimizer.apply_gradients(zip(grads, [self.linear.w, self.linear.b]))

        final_loss = tf.reduce_mean(
            0.5 * tf.square(y - self.linear(self.basis_expansion(x)))
        )

        self.assertLess(final_loss, initial_loss)

    @parameterized.expand([(2,), (4,), (6,), (8,)])
    def test_train_with_different_range_end(self, range_end):
        self.config["RANGE_END"] = range_end
        self.basis_expansion = BasisExpansion(
            self.config["NUM_BASIS_FUNCTIONS"], range_end
        )

        x = tf.constant([[0.0], [1.0], [2.0]], dtype=tf.float32)
        y = tf.constant([[0.0], [1.0], [2.0]], dtype=tf.float32)

        optimizer = tf.optimizers.SGD(self.config["STEP_SIZE"])

        # Initial loss
        initial_loss = tf.reduce_mean(
            0.5 * tf.square(y - self.linear(self.basis_expansion(x)))
        )

        # Training
        with tf.GradientTape() as tape:
            tape.watch([self.linear.w, self.linear.b])
            y_hat = self.linear(self.basis_expansion(x))
            loss = tf.reduce_mean(0.5 * tf.square(y - y_hat))

        grads = tape.gradient(loss, [self.linear.w, self.linear.b])
        optimizer.apply_gradients(zip(grads, [self.linear.w, self.linear.b]))

        # Final loss
        final_loss = tf.reduce_mean(
            0.5 * tf.square(y - self.linear(self.basis_expansion(x)))
        )

        self.assertLess(final_loss, initial_loss)

    def test_optimizer(self):
        with tf.GradientTape() as tape:
            tape.watch(self.linear.w)
            y_hat = self.linear(
                self.basis_expansion(tf.constant([[0.0], [1.0], [2.0]]))
            )
        grad = tape.gradient(y_hat, self.linear.w)
        self.assertIsNotNone(grad)

    def test_r_squared(self):
        tf.random.set_seed(12345)
        x = tf.constant([[0.0], [1.0], [2.0]], dtype=tf.float32)
        y = tf.constant([[0.0], [1.0], [2.0]], dtype=tf.float32)
        y_hat = self.linear(self.basis_expansion(x))

        # Verifying if y_hat and y shapes are as expected
        self.assertEqual(y_hat.shape, y.shape)

        y_bar = tf.reduce_mean(y)
        ss_tot = tf.reduce_sum(tf.square(y - y_bar))
        ss_res = tf.reduce_sum(tf.square(y - y_hat))
        r_squared = 1 - ss_res / ss_tot

        self.assertAlmostEqual(float(r_squared.numpy()), -1.8298792839050293)


if __name__ == "__main__":
    unittest.main()

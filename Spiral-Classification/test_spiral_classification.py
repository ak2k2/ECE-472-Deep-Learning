import os
import unittest

import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # disable tensorflow warnings
import tensorflow as tf
from generate_spirals import generate_spiral_data
from linear_module import Linear
from mlp_module import MLP
from spiral_classification import log_cross_entropy_loss, l2_regularization


class TestGenerateSpirals(unittest.TestCase):
    def test_generate_spirals(self):
        N = 200
        K = 3
        x_data, y_data = generate_spiral_data(N, K, 0)
        # Check shapes
        self.assertEqual(x_data.shape, (2 * N, 2))
        self.assertEqual(y_data.shape, (2 * N, 1))


class TestLogCrossEntropyLoss(unittest.TestCase):
    def test_log_cross_entropy_loss(self):
        y_true = [[0.0], [1.0], [0.0], [0.0]]
        y_pred = [[-18.6], [0.51], [2.94], [-12.8]]
        y_true = tf.constant(y_true)
        y_pred = tf.constant(y_pred)

        loss = log_cross_entropy_loss(
            y_true, y_pred
        )  # logits are set to True by default

        # Convert TensorFlow tensor to Python float
        loss_value = float(loss.numpy())

        self.assertAlmostEqual(loss_value, 4.153932571411133)

    def test_l2_regularization(self):
        layers = [
            Linear(1, 1, 0),
            Linear(1, 1, 0),
            Linear(1, 1, 0),
        ]
        lambda_reg = 0.1
        reg_loss = l2_regularization(layers, lambda_reg)
        self.assertAlmostEqual(reg_loss.numpy(), 1.1166232)

    def test_linear(self):
        input_dim = 2
        output_dim = 3
        seed = 0
        linear = Linear(input_dim, output_dim, seed)
        x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        y = linear(x)
        self.assertEqual(y.shape, (2, 3))

    def test_mlp(self):
        mlp = MLP(
            num_inputs=2,
            num_outputs=3,
            num_hidden_layers=2,
            hidden_layer_width=4,
            hidden_activation=tf.nn.relu,
            output_activation=tf.nn.sigmoid,
            seed=0,
        )
        x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        y = mlp(x)
        self.assertEqual(y.shape, (2, 3))
        # make sure the output is in [0, 1]
        self.assertTrue(tf.reduce_all(y >= 0.0))
        self.assertTrue(tf.reduce_all(y <= 1.0))


if __name__ == "__main__":
    unittest.main()

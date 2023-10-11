import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # disable tensorflow warnings
import tensorflow as tf

"""
Linear class from Spiral Classification Assignment: Tested in Spiral-Classification/test_spiral_classification.py
"""


class Linear(tf.Module):
    def __init__(self, input_dim, output_dim, seed=None):
        initializer = tf.initializers.HeNormal(seed=seed)
        self.w = tf.Variable(
            initializer([input_dim, output_dim]), dtype=tf.float32, trainable=True
        )  # Ensure the data type is float32
        self.b = tf.Variable(
            tf.zeros([output_dim]), dtype=tf.float32, trainable=True
        )  # Ensure the data type is float32

    def __call__(self, x):
        x = tf.cast(x, tf.float32)  # Cast the input to float32
        return tf.matmul(x, self.w) + self.b

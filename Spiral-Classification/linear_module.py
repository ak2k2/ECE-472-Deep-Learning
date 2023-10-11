import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # disable tensorflow warnings
import tensorflow as tf


class Linear(tf.Module):
    def __init__(self, input_dim, output_dim, seed=None):
        initializer = tf.initializers.GlorotNormal(seed)
        self.w = tf.Variable(initializer([input_dim, output_dim]), trainable=True)
        self.b = tf.Variable(tf.zeros([output_dim]), trainable=True)

    def __call__(self, x):
        return tf.matmul(x, self.w) + self.b

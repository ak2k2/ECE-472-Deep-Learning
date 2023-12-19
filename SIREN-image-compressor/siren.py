import math

import numpy as np
import tensorflow as tf

# Implicit Neural Representations with Periodic Activation Functions: https://doi.org/10.48550/arXiv.2006.09661


class Linear(tf.Module):
    def __init__(self, in_features, out_features, seed=None):
        initializer = tf.initializers.GlorotNormal(seed=seed)
        # Explicitly set data type to tf.float32
        self.w = tf.Variable(
            initializer([in_features, out_features]), dtype=tf.float32, trainable=True
        )
        self.b = tf.Variable(tf.zeros([out_features], dtype=tf.float32), trainable=True)

    def __call__(self, x):
        x = tf.cast(x, tf.float32)
        return tf.matmul(x, self.w) + self.b


class SirenNet(tf.Module):
    def __init__(
        self, in_features, hidden_features, hidden_layers, out_features, w0_initial=30.0
    ):
        self.layers = []
        for i in range(hidden_layers):
            in_dim = in_features if i == 0 else hidden_features
            out_dim = out_features if i == hidden_layers - 1 else hidden_features
            w0 = w0_initial if i == 0 else 1.0
            layer = Linear(in_dim, out_dim)
            self.layers.append((layer, w0))

    def __call__(self, x):
        for layer, w0 in self.layers[:-1]:
            x = tf.math.sin(w0 * layer(x))
        final_layer, _ = self.layers[-1]
        return final_layer(x)  # No activation in the last layer

    def initialize_weights(self):
        # Custom initialization as described in the paper
        for layer, w0 in self.layers:
            layer.w.assign(
                tf.random.uniform(
                    layer.w.shape,
                    minval=-1 / math.sqrt(layer.w.shape[0]),
                    maxval=1 / math.sqrt(layer.w.shape[0]),
                )
            )
            layer.w.assign(layer.w * w0)

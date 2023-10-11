import os
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

from conv2d import Conv2d

tf.random.set_seed(0)
np.random.seed(0)


class Classifier(tf.Module):
    def __init__(
        self,
        input_depth,
        conv_layer_depths,
        conv_kernel_sizes,
        fc_layer_sizes,
        num_classes,
        dropout_rate,
    ):
        self.conv_layers = []
        self.fc_layers = []
        self.input_depth = input_depth
        self.dropout_rate = dropout_rate

        for depth, kernel_size in zip(conv_layer_depths, conv_kernel_sizes):
            self.conv_layers.append(Conv2d(self.input_depth, depth, kernel_size))
            self.input_depth = depth  # Update input depth for next layer

        image_size = 28  # initial image size
        for _ in conv_layer_depths:
            image_size = image_size // 2
        flattened_size = image_size * image_size * self.input_depth

        dummy_input = tf.constant(
            tf.random.normal([1, 28, 28, input_depth]), dtype=tf.float32
        )
        for conv_layer in self.conv_layers:
            dummy_input = conv_layer.forward(dummy_input)
            dummy_input = tf.nn.max_pool(
                dummy_input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME"
            )

        flattened_size = tf.reduce_prod(dummy_input.shape[1:]).numpy()

        input_size = flattened_size
        for size in fc_layer_sizes:
            # Using He Initialization for fully connected layers as well
            stddev = np.sqrt(2.0 / input_size)
            self.fc_layers.append(
                tf.Variable(tf.random.normal([input_size, size], stddev=stddev))
            )
            input_size = size

        # Output layer
        stddev = np.sqrt(2.0 / input_size)
        self.output_layer = tf.Variable(
            tf.random.normal([input_size, num_classes], stddev=stddev)
        )

    def forward(self, x):
        for conv_layer in self.conv_layers:
            x = conv_layer.forward(x)
            x = tf.nn.max_pool(
                x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME"
            )  # Max pooling
        # Flatten the tensor
        x = tf.reshape(x, [-1, tf.reduce_prod(x.shape[1:])])

        # Fully connected layers
        for fc_layer in self.fc_layers:
            x = tf.matmul(x, fc_layer)
            x = tf.nn.leaky_relu(x, alpha=0.3)
            x = tf.nn.dropout(x, rate=self.dropout_rate)

        # Output layer
        x = tf.matmul(x, self.output_layer)

        return x  # Logits

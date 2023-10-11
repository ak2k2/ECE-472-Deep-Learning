import os
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

# set ranom seed
tf.random.set_seed(0)
np.random.seed(0)


class Conv2d(tf.Module):
    def __init__(self, input_channels, output_channels, kernel_size):
        # Initialize weights using He Initialization
        stddev = np.sqrt(2.0 / (kernel_size[0] * kernel_size[1] * input_channels))
        self.weights = tf.Variable(
            tf.random.normal(
                [kernel_size[0], kernel_size[1], input_channels, output_channels],
                stddev=stddev,
            )
        )

        self.biases = tf.Variable(tf.zeros([output_channels]))

    def forward(self, x):
        x = tf.nn.conv2d(x, self.weights, strides=[1, 1, 1, 1], padding="SAME")
        x = tf.nn.bias_add(x, self.biases)
        return tf.nn.relu(x)

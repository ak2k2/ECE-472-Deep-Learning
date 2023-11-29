import os
import numpy as np
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Set random seed for reproducibility
tf.random.set_seed(0)
np.random.seed(0)


class Conv2d(tf.Module):
    def __init__(self, input_channels, output_channels, kernel_size):
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


class GroupNorm(tf.Module):
    def __init__(self, num_channels, num_groups):
        self.num_groups = num_groups
        self.gamma = tf.Variable(tf.ones([num_channels]), trainable=True)
        self.beta = tf.Variable(tf.zeros([num_channels]), trainable=True)

    def forward(self, x):
        N, H, W, C = x.shape
        x = tf.reshape(x, [N, H, W, self.num_groups, C // self.num_groups])
        mean, variance = tf.nn.moments(x, axes=[1, 2, 4], keepdims=True)
        x = (x - mean) / tf.sqrt(variance + 1e-5)
        x = tf.reshape(x, [N, H, W, C])
        return x * self.gamma + self.beta


class ResidualBlock(tf.Module):
    def __init__(self, input_channels, output_channels, kernel_size, num_groups):
        self.conv1 = Conv2d(input_channels, output_channels, kernel_size)
        self.groupnorm1 = GroupNorm(output_channels, num_groups)
        self.conv2 = Conv2d(output_channels, output_channels, kernel_size)
        self.groupnorm2 = GroupNorm(output_channels, num_groups)

        # Adjust dimensions if needed
        if input_channels != output_channels:
            self.adjust_dimensions = Conv2d(input_channels, output_channels, [1, 1])
        else:
            self.adjust_dimensions = None

    def forward(self, x):
        identity = x

        out = self.conv1.forward(x)
        out = self.groupnorm1.forward(out)
        out = tf.nn.relu(out)

        out = self.conv2.forward(out)
        out = self.groupnorm2.forward(out)

        if self.adjust_dimensions is not None:
            identity = self.adjust_dimensions.forward(x)

        out += identity
        return tf.nn.relu(out)


class Classifier(tf.Module):
    def __init__(self, num_classes):
        self.initial_conv = Conv2d(
            3, 64, [3, 3]
        )  # Adjust for CIFAR-10's 3 color channels
        self.res_block1 = ResidualBlock(64, 128, [3, 3], num_groups=4)
        self.res_block2 = ResidualBlock(128, 256, [3, 3], num_groups=4)
        self.res_block3 = ResidualBlock(256, 512, [3, 3], num_groups=4)
        self.global_pool = lambda x: tf.reduce_mean(x, [1, 2])  # Global Average Pooling
        self.fc_layer = tf.Variable(tf.random.normal([512, num_classes]))

    def forward(self, x):
        x = self.initial_conv.forward(x)
        x = self.res_block1.forward(x)
        x = self.res_block2.forward(x)
        x = self.res_block3.forward(x)
        x = self.global_pool(x)
        x = tf.matmul(x, self.fc_layer)
        return x

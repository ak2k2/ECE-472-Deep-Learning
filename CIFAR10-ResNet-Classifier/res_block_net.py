import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa


class Linear(tf.Module):
    """
    The weight matrix self.w is initialized with the shape [input_dim, output_dim]

    input_dim is the number of features of the input (determined by the output of the global average pooling layer)
    output_dim is the number of classes (10 in your case).
    """

    def __init__(self, input_dim, output_dim):
        initializer = tf.initializers.HeUniform()
        self.w = tf.Variable(initializer([input_dim, output_dim]), trainable=True)
        self.b = tf.Variable(tf.zeros([output_dim]), trainable=True)

    def __call__(self, x):
        return tf.matmul(x, self.w) + self.b  # y = w^T * x + b


class CustomConv2D(tf.Module):
    def __init__(self, filters, kernel_size, activation=None):
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.initialized = False

    def __call__(self, x):
        if not self.initialized:
            self.kernel = tf.Variable(
                tf.random.normal(
                    [self.kernel_size, self.kernel_size, x.shape[-1], self.filters]
                )
            )
            self.bias = tf.Variable(tf.zeros([self.filters]))
            self.initialized = True

        x = tf.nn.conv2d(x, self.kernel, strides=[1, 1, 1, 1], padding="SAME")
        x = tf.nn.bias_add(x, self.bias)
        if self.activation:
            x = self.activation(x)
        return x


class CustomGroupNorm(
    tf.Module
):  # TODO: Fix this class and swap out the tfa.layers.GroupNormalization class in the CustomResidualBlock class below.
    def __init__(self, num_channels, num_groups, epsilon=1e-3):
        super(CustomGroupNorm, self).__init__()
        self.num_channels = num_channels
        self.num_groups = num_groups
        self.epsilon = epsilon
        self.gamma = tf.Variable(tf.ones([1, 1, 1, num_channels]), trainable=True)
        self.beta = tf.Variable(tf.zeros([1, 1, 1, num_channels]), trainable=True)

    def __call__(self, x):
        N, H, W, C = x.shape

        if C % self.num_groups != 0:
            raise ValueError("Number of channels must be divisible by number of groups")

        # Reshape x to [N, H, W, num_groups, num_channels // num_groups]
        x = tf.reshape(x, [N, H, W, self.num_groups, C // self.num_groups])

        # Calculate mean and variance
        mean, variance = tf.nn.moments(x, axes=[1, 2, 4], keepdims=True)

        # Normalize x
        x = (x - mean) / tf.sqrt(variance + self.epsilon)

        # Reshape gamma and beta to align with the group structure
        gamma_reshaped = tf.reshape(
            self.gamma, [1, 1, 1, self.num_groups, C // self.num_groups]
        )
        beta_reshaped = tf.reshape(
            self.beta, [1, 1, 1, self.num_groups, C // self.num_groups]
        )

        # Apply gamma and beta
        x = x * gamma_reshaped + beta_reshaped

        # Reshape back to original shape
        return tf.reshape(x, [N, H, W, C])


class CustomResidualBlock(tf.Module):
    def __init__(self, input_channels, output_channels, kernel_size, num_groups):
        self.conv1 = CustomConv2D(input_channels, kernel_size, activation=tf.nn.relu)
        self.groupnorm1 = tfa.layers.GroupNormalization(
            groups=num_groups, axis=-1
        )  # this is NOT the CustomGroupNorm class above. This is a built-in class from tensorflow-addons. My CustomGroupNorm class does not cast beta and gamma correctly.
        self.conv2 = CustomConv2D(output_channels, kernel_size, activation=None)
        self.groupnorm2 = tfa.layers.GroupNormalization(groups=num_groups, axis=-1)
        self.relu = tf.nn.relu

        if input_channels != output_channels:
            self.dimension_adjust = CustomConv2D(output_channels, 1, activation=None)
        else:
            self.dimension_adjust = None

    def __call__(self, x):
        identity = x
        out = self.conv1(x)
        out = self.groupnorm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.groupnorm2(out)

        if self.dimension_adjust:
            identity = self.dimension_adjust(identity)

        out += identity
        return self.relu(out)


class Classifier(tf.Module):
    def __init__(self, num_classes):
        self.res_block1 = CustomResidualBlock(32, 64, 3, 4)
        self.res_block2 = CustomResidualBlock(64, 128, 3, 4)
        self.global_pool = lambda x: tf.reduce_mean(x, [1, 2])
        # Adjusting for the output shape of the global pooling layer
        self.fc = Linear(128, num_classes)

    def __call__(self, x):
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.global_pool(x)
        # Flattening before passing through the Linear layer
        x = tf.reshape(x, [x.shape[0], -1])
        return self.fc(x)


# Tests
def test_custom_conv2d_initialization_and_activation():
    conv_layer = CustomConv2D(filters=32, kernel_size=3, activation=tf.nn.relu)
    x = tf.random.normal([1, 64, 64, 3])

    conv_layer(x)  # Call to initialize variables
    assert conv_layer.kernel.shape == (
        3,
        3,
        3,
        32,
    ), f"Kernel shape incorrect, got {conv_layer.kernel.shape}"
    assert tf.reduce_all(tf.equal(conv_layer.bias, 0)), "Bias not initialized to zeros"

    # Test activation function
    output = conv_layer(x)
    assert tf.reduce_all(output >= 0), "ReLU not applied correctly"


def test_custom_residual_block():
    block = CustomResidualBlock(32, 64, 3, 4)
    x = tf.random.normal([1, 64, 64, 32])

    # Test the block's functionality
    output = block(x)
    assert output.shape == (
        1,
        64,
        64,
        64,
    ), "Output shape of Residual Block is incorrect"

    # Test dimension adjustment
    assert output.shape[-1] == 64, "Dimension adjustment in residual block failed"


def test_global_pooling_and_linear_layer():
    model = Classifier(num_classes=10)

    # The number of channels should match the input channels of the first residual block
    x = tf.random.normal(
        [1, 64, 64, 32]
    )  # Assuming the first residual block takes 32 channels as input

    output = model(x)
    assert output.shape == (1, 10), "Output shape of the model is incorrect"


def test_forward_pass_through_classifier():
    model = Classifier(num_classes=10)
    x = tf.random.normal([1, 64, 64, 32])

    output = model(x)
    assert output.shape == (1, 10), "Forward pass through classifier failed"


if __name__ == "__main__":
    test_custom_conv2d_initialization_and_activation()
    test_custom_residual_block()
    test_global_pooling_and_linear_layer()
    test_forward_pass_through_classifier()
    print("All tests passed successfully!")

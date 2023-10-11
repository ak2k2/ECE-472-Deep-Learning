import os
import sys
import unittest

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from conv2d import (  # Replace 'your_module' with the actual module name where Conv2d is defined
    Conv2d,
)


class TestConv2d(unittest.TestCase):
    def test_output_shape(self):
        # Initialize Conv2d layer
        conv_layer = Conv2d(input_channels=1, output_channels=32, kernel_size=(3, 3))

        # A dummy input
        dummy_input = tf.constant(
            tf.random.normal([1, 28, 28, 1]), dtype=tf.float32
        )  # 1 sample, 28x28 image, 1 channel

        output = conv_layer.forward(dummy_input)

        # Confirm the output shape
        self.assertEqual(output.shape, (1, 28, 28, 32))


if __name__ == "__main__":
    unittest.main()

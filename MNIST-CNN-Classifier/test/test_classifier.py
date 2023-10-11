import os
import sys
import unittest

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classifier import Classifier


class TestClassifier(unittest.TestCase):
    def setUp(self):
        self.classifier = Classifier(
            input_depth=1,
            conv_layer_depths=[32, 64],
            conv_kernel_sizes=[(3, 3), (3, 3)],
            fc_layer_sizes=[128],
            num_classes=10,
        )
        self.dummy_input = tf.constant(
            tf.random.normal([1, 28, 28, 1]), dtype=tf.float32
        )  # 1 sample, 28x28 image, 1 channel

    def test_forward_output_shape(self):
        output = self.classifier.forward(self.dummy_input)
        self.assertEqual(output.shape, (1, 10))

    def test_forward_output_type(self):
        output = self.classifier.forward(self.dummy_input)
        self.assertTrue(tf.is_tensor(output))

    def test_forward_output_dtype(self):
        output = self.classifier.forward(self.dummy_input)
        self.assertEqual(output.dtype, tf.float32)


if __name__ == "__main__":
    unittest.main()

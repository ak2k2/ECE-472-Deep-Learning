import os
import sys
import unittest

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from read_mnist_data_from_ubytes import get_shuffled_mnist_data


class TestLoadMNISTData(unittest.TestCase):
    def setUp(self):
        (
            self.train_labels,
            self.train_images,
            self.val_labels,
            self.val_images,
            self.test_labels,
            self.test_images,
        ) = get_shuffled_mnist_data()

    def test_data_shapes(self):
        # Confirm the shapes are correct
        self.assertEqual(self.train_labels.shape, (54000,))
        self.assertEqual(self.train_images.shape, (54000, 28, 28, 1))
        self.assertEqual(self.val_labels.shape, (6000,))
        self.assertEqual(self.val_images.shape, (6000, 28, 28, 1))
        self.assertEqual(self.test_labels.shape, (10000,))
        self.assertEqual(self.test_images.shape, (10000, 28, 28, 1))

    def test_data_types(self):
        # Confirm the data types are correct
        self.assertTrue(tf.is_tensor(self.train_labels))
        self.assertTrue(tf.is_tensor(self.train_images))
        self.assertTrue(tf.is_tensor(self.val_labels))
        self.assertTrue(tf.is_tensor(self.val_images))
        self.assertTrue(tf.is_tensor(self.test_labels))
        self.assertTrue(tf.is_tensor(self.test_images))

        self.assertEqual(self.train_labels.dtype, tf.int64)
        self.assertEqual(self.train_images.dtype, tf.float32)
        self.assertEqual(self.val_labels.dtype, tf.int64)
        self.assertEqual(self.val_images.dtype, tf.float32)
        self.assertEqual(self.test_labels.dtype, tf.int64)
        self.assertEqual(self.test_images.dtype, tf.float32)


if __name__ == "__main__":
    unittest.main()

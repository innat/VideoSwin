import tensorflow as tf
from absl.testing import parameterized
from keras import ops


class TestCase(tf.test.TestCase, parameterized.TestCase):
    """Base test case class. (Copied from KerasCV)."""

    def assertAllClose(self, x1, x2, atol=1e-6, rtol=1e-6, msg=None):
        x1 = tf.nest.map_structure(convert_to_numpy, x1)
        x2 = tf.nest.map_structure(convert_to_numpy, x2)
        super().assertAllClose(x1, x2, atol=atol, rtol=rtol, msg=msg)

    def assertAllEqual(self, x1, x2, msg=None):
        x1 = tf.nest.map_structure(convert_to_numpy, x1)
        x2 = tf.nest.map_structure(convert_to_numpy, x2)
        super().assertAllEqual(x1, x2, msg=msg)


def convert_to_numpy(x):
    if ops.is_tensor(x) and not isinstance(x, tf.RaggedTensor):
        return ops.convert_to_numpy(x)
    return x

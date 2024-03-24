import os

import keras
import numpy as np
import pytest
import tensorflow as tf
from absl.testing import parameterized
from base import TestCase
from keras import ops

from videoswin.model import VideoSwinT


class VideoClassifierTest(TestCase):
    def setUp(self):
        self.input_batch = np.ones(shape=(10, 8, 224, 224, 3))
        self.dataset = tf.data.Dataset.from_tensor_slices(
            (self.input_batch, tf.one_hot(tf.ones((10,), dtype="int32"), 10))
        ).batch(4)

    def test_valid_call(self):
        input_batch = np.ones(shape=(2, 8, 256, 256, 3))
        model = VideoSwinT(
            input_shape=(8, 256, 256, 3),
            include_rescaling=False,
            num_classes=10,
        )
        model(input_batch)

    def test_valid_call_non_square_shape(self):
        input_batch = np.ones(shape=(2, 8, 224, 256, 3))
        model = VideoSwinT(
            input_shape=(8, 224, 256, 3),
            include_rescaling=False,
            num_classes=10,
        )
        model(input_batch)

    @parameterized.named_parameters(
        ("jit_compile_false", False), ("jit_compile_true", True)
    )
    @pytest.mark.large  # Fit is slow, so mark these large.
    def test_classifier_fit(self, jit_compile):
        model = VideoSwinT(
            input_shape=(8, 224, 224, 3),
            include_rescaling=True,
            num_classes=10,
        )
        model.compile(
            loss="categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"],
            jit_compile=jit_compile,
        )
        model.fit(self.dataset)

    @parameterized.named_parameters(("avg_pooling", "avg"), ("max_pooling", "max"))
    def test_pooling_arg_call(self, pooling):
        input_batch = np.ones(shape=(2, 8, 224, 224, 3))
        model = VideoSwinT(
            input_shape=(8, 224, 224, 3),
            include_rescaling=True,
            num_classes=10,
            pooling=pooling,
        )
        model(input_batch)

    @pytest.mark.large  # Saving is slow, so mark these large.
    def test_saved_model(self):
        self.skipTest("Skipping test save with keras format.")
        model = VideoSwinT(
            input_shape=(8, 224, 224, 3),
            include_rescaling=False,
            num_classes=10,
        )
        model_output = model(self.input_batch)
        save_path = os.path.join(self.get_temp_dir(), "video_classifier.keras")
        model.save(save_path)
        restored_model = keras.models.load_model(save_path)

        # Check we got the real object back.
        self.assertIsInstance(restored_model, VideoSwinT)

        # Check that output matches.
        restored_output = restored_model(self.input_batch)
        self.assertAllClose(
            ops.convert_to_numpy(model_output),
            ops.convert_to_numpy(restored_output),
        )

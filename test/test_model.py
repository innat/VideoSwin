# noqa: E501

import os

import numpy as np
import pytest
from absl.testing import parameterized
from base import TestCase
import tensorflow as tf
from keras import ops
import keras
from videoswin.model import VideoSwinBackbone
from videoswin.model import VideoSwinT


class TestVideoSwinSBackbone(TestCase):

    @pytest.mark.large
    def test_call(self):
        model = VideoSwinBackbone(  # TODO: replace with aliases
            include_rescaling=True, input_shape=(8, 256, 256, 3)
        )
        x = np.ones((1, 8, 256, 256, 3))
        x_out = ops.convert_to_numpy(model(x))
        num_parameters = sum(
            np.prod(tuple(x.shape)) for x in model.trainable_variables
        )
        self.assertEqual(x_out.shape, (1, 4, 8, 8, 768))
        self.assertEqual(num_parameters, 27_663_894)

    @pytest.mark.extra_large
    def teat_save(self):
        # saving test
        model = VideoSwinBackbone(include_rescaling=False)
        x = np.ones((1, 32, 224, 224, 3))
        x_out = ops.convert_to_numpy(model(x))
        path = os.path.join(self.get_temp_dir(), "model.keras")
        model.save(path)
        loaded_model = keras.saving.load_model(path)
        x_out_loaded = ops.convert_to_numpy(loaded_model(x))
        self.assertAllClose(x_out, x_out_loaded)

    @pytest.mark.extra_large
    def test_fit(self):
        model = VideoSwinBackbone(include_rescaling=False)
        x = np.ones((1, 32, 224, 224, 3))
        y = np.zeros((1, 16, 7, 7, 768))
        model.compile(optimizer="adam", loss="mse", metrics=["mse"])
        model.fit(x, y, epochs=1)

    @pytest.mark.extra_large
    def test_can_run_in_mixed_precision(self):
        keras.mixed_precision.set_global_policy("mixed_float16")
        model = VideoSwinBackbone(
            include_rescaling=False, input_shape=(8, 224, 224, 3)
        )
        x = np.ones((1, 8, 224, 224, 3))
        y = np.zeros((1, 4, 7, 7, 768))
        model.compile(optimizer="adam", loss="mse", metrics=["mse"])
        model.fit(x, y, epochs=1)

    @pytest.mark.extra_large
    def test_can_run_on_gray_video(self):
        model = VideoSwinBackbone(
            include_rescaling=False,
            input_shape=(96, 96, 96, 1),
            window_size=[6, 6, 6],
        )
        x = np.ones((1, 96, 96, 96, 1))
        y = np.zeros((1, 48, 3, 3, 768))
        model.compile(optimizer="adam", loss="mse", metrics=["mse"])
        model.fit(x, y, epochs=1)


class VideoClassifierTest(TestCase):
    def setUp(self):
        self.input_batch = np.ones(shape=(10, 8, 224, 224, 3))
        self.dataset = tf.data.Dataset.from_tensor_slices(
            (self.input_batch, tf.one_hot(tf.ones((10,), dtype="int32"), 10))
        ).batch(4)

    def test_valid_call(self):
        model = VideoSwinT(
            backbone=VideoSwinBackbone(
                input_shape=(8, 224, 224, 3), include_rescaling=False
            ),
            num_classes=10,
        )
        model(self.input_batch)

    @parameterized.named_parameters(
        ("jit_compile_false", False), ("jit_compile_true", True)
    )
    @pytest.mark.large  # Fit is slow, so mark these large.
    def test_classifier_fit(self, jit_compile):
        model = VideoSwinT(
            backbone=VideoSwinBackbone(
                input_shape=(8, 224, 224, 3), include_rescaling=True
            ),
            num_classes=10,
        )
        model.compile(
            loss="categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"],
            jit_compile=jit_compile,
        )
        model.fit(self.dataset)

    @parameterized.named_parameters(
        ("avg_pooling", "avg"), ("max_pooling", "max")
    )
    def test_pooling_arg_call(self, pooling):
        model = VideoSwinT(
            backbone=VideoSwinBackbone(
                input_shape=(8, 224, 224, 3), include_rescaling=True
            ),
            num_classes=10,
            pooling=pooling,
        )
        model(self.input_batch)

    @pytest.mark.large  # Saving is slow, so mark these large.
    def test_saved_model(self):
        model = VideoSwinT(
            backbone=VideoSwinBackbone(
                input_shape=(8, 224, 224, 3), include_rescaling=False
            ),
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

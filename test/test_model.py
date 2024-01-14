# noqa: E501

import os
import pytest
import numpy as np
from absl.testing import parameterized

from keras import ops
from base import TestCase

from videoswin import VideoSwinT


class TestVideoSwin(TestCase):
    def setUp(self):
        self.input_batch = ops.ones(shape=(1, 32, 224, 224, 3))
        self.expected_shapes = {
            "BasicLayer1_attention_maps": (128, 3, 392, 392),
            "BasicLayer2_attention_maps": (32, 6, 392, 392),
            "BasicLayer3_attention_maps": (8, 12, 392, 392),
            "BasicLayer4_attention_maps": (2, 24, 392, 392),
        }

    @parameterized.named_parameters(
        {"testcase_name": "num_classes_400", "num_classes": 400},
        {"testcase_name": "num_classes_174", "num_classes": 174},
        {"testcase_name": "num_classes_101", "num_classes": 101},
    )
    @pytest.mark.large
    def test_call(self, num_classes):
        # build the model and run
        model = VideoSwinT(num_classes=num_classes)
        x = self.input_batch
        x_out, attention_maps_dict = model(x, return_attention_maps=True)

        # compute params
        num_parameters = sum(np.prod(tuple(x.shape)) for x in model.trainable_variables)

        # presets
        expected_parameters = {
            400: 28_158_070,  # Kinetics-400
            174: 27_984_276,  # Something-Something-V2
            101: 27_928_139,  # UCF101
        }.get(num_classes)

        # assert test
        self.assertEqual(x_out.shape, (1, num_classes))
        self.assertEqual(num_parameters, expected_parameters)
        for key, expected_shape in self.expected_shapes.items():
            self.assertEqual(attention_maps_dict[key].shape, expected_shape)

    @parameterized.named_parameters(
        {"testcase_name": "num_classes_400", "num_classes": 400},
        {"testcase_name": "num_classes_174", "num_classes": 174},
        {"testcase_name": "num_classes_101", "num_classes": 101},
    )
    @pytest.mark.extra_large
    def test_save(self, num_classes):
        # saving test
        x = self.input_batch

        # build the model and run and save weights
        model = VideoSwinT(num_classes=num_classes)
        x_out = model(x)
        path = os.path.join(self.get_temp_dir(), "model.weights.h5")
        model.save_weights(path)

        # load the saved model
        loaded_model = VideoSwinT(num_classes=num_classes)
        loaded_model(x)
        loaded_model.load_weights(path)
        x_out_loaded = ops.convert_to_numpy(loaded_model(x))

        # assert test
        self.assertAllClose(x_out, x_out_loaded)

    @parameterized.named_parameters(
        {"testcase_name": "input_shape_8x96x96", "input_shape": (8, 96, 96, 3)},
        {"testcase_name": "input_shape_8x224x224", "input_shape": (8, 224, 224, 3)},
        {"testcase_name": "input_shape_16x96x96", "input_shape": (16, 96, 96, 3)},
        {"testcase_name": "input_shape_16x312x312", "input_shape": (16, 312, 312, 3)},
        {"testcase_name": "input_shape_32x224x224", "input_shape": (32, 224, 224, 3)},
    )
    @pytest.mark.extra_large
    def test_call_with_variable_shape(self, input_shape):
        model = VideoSwinT(num_classes=400)
        x_inp = ops.ones(shape=(1, *input_shape))
        x_out = model(x_inp)
        self.assertEqual(x_out.shape, (1, 400))

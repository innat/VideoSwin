import os
import warnings

warnings.simplefilter(action="ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import keras

from videoswin.backbone import VideoSwinBackbone


@keras.utils.register_keras_serializable(package="swin.transformer.tiny.3d")
def VideoSwinT(
    input_shape=(32, 224, 224, 3),
    num_classes=400,
    pooling="avg",
    activation="softmax",
    window_size=(8, 7, 7),
    embed_size=96,
    depths=[2, 2, 6, 2],
    num_heads=[3, 6, 12, 24],
    include_rescaling=False,
    include_top=True,
    **kwargs,
):
    if pooling == "avg":
        pooling_layer = keras.layers.GlobalAveragePooling3D(name="avg_pool")
    elif pooling == "max":
        pooling_layer = keras.layers.GlobalMaxPooling3D(name="max_pool")
    else:
        raise ValueError(f'`pooling` must be one of "avg", "max". Received: {pooling}.')

    backbone = VideoSwinBackbone(
        input_shape=input_shape,
        window_size=window_size,
        embed_dim=embed_size,
        depths=depths,
        num_heads=num_heads,
        include_rescaling=include_rescaling,
        name="VideoSwinTBackbone",
    )

    if not include_top:
        return backbone

    inputs = backbone.input
    x = backbone(inputs)
    x = pooling_layer(x)
    outputs = keras.layers.Dense(
        num_classes,
        activation=activation,
        name="predictions",
        dtype="float32",
    )(x)

    return keras.Model(inputs=inputs, outputs=outputs, name="VideoSwinT", **kwargs)


# @keras.utils.register_keras_serializable(package="swin.transformer.tiny.3d")
# class VideoSwinT(keras.Model):
#     def __init__(
#         self,
#         input_shape=(32, 224, 224, 3),
#         num_classes=400,
#         pooling="avg",
#         activation="softmax",
#         window_size=(8,7,7),
#         embed_size=96,
#         depths=[2, 2, 6, 2],
#         num_heads=[3, 6, 12, 24],
#         include_rescaling=False,
#         include_top=True,
#         **kwargs,
#     ):

#         if pooling == "avg":
#             pooling_layer = keras.layers.GlobalAveragePooling3D(name="avg_pool")
#         elif pooling == "max":
#             pooling_layer = keras.layers.GlobalMaxPooling3D(name="max_pool")
#         else:
#             raise ValueError(
#                 f'`pooling` must be one of "avg", "max". Received: {pooling}.'
#             )

#         backbone = VideoSwinBackbone(
#             input_shape=input_shape,
#             window_size=window_size,
#             embed_dim=embed_size,
#             depths=depths,
#             num_heads=num_heads,
#             include_rescaling=include_rescaling,
#         )

#         if not include_top:
#             return backbone

#         inputs = backbone.input
#         x = backbone(inputs)
#         x = pooling_layer(x)
#         outputs = keras.layers.Dense(
#             num_classes,
#             activation=activation,
#             name="predictions",
#             dtype="float32",
#         )(x)
#         super().__init__(inputs=inputs, outputs=outputs, name='VideoSwinT', **kwargs)
#         self.window_size = window_size
#         self.num_classes = num_classes
#         self.pooling = pooling
#         self.activation = activation
#         self.embed_size = embed_size
#         self.depths = depths
#         self.num_heads = num_heads
#         self.include_rescaling = include_rescaling
#         self.include_top = include_top

#     def get_config(self):
#         config = {
#             "input_shape": self.input_shape[1:],
#             "window_size": self.window_size,
#             "num_classes": self.num_classes,
#             "pooling": self.pooling,
#             "activation": self.activation,
#             "embed_size": self.embed_size,
#             "depths": self.depths,
#             "num_heads": self.num_heads,
#             "include_rescaling": self.include_rescaling,
#             "include_top": self.include_top,
#         }
#         return config


@keras.utils.register_keras_serializable(package="swin.transformer.small.3d")
class VideoSwinS(keras.Model):
    def __init__(
        self,
        input_shape=(32, 224, 224, 3),
        num_classes=400,
        pooling="avg",
        activation="softmax",
        window_size=(8, 7, 7),
        embed_size=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        include_rescaling=False,
        include_top=True,
        **kwargs,
    ):

        if pooling == "avg":
            pooling_layer = keras.layers.GlobalAveragePooling3D(name="avg_pool")
        elif pooling == "max":
            pooling_layer = keras.layers.GlobalMaxPooling3D(name="max_pool")
        else:
            raise ValueError(
                f'`pooling` must be one of "avg", "max". Received: {pooling}.'
            )

        backbone = VideoSwinBackbone(
            input_shape=input_shape,
            window_size=window_size,
            embed_dim=embed_size,
            depths=depths,
            num_heads=num_heads,
            include_rescaling=include_rescaling,
        )

        if not include_top:
            return backbone

        inputs = backbone.input
        x = backbone(inputs)
        x = pooling_layer(x)
        outputs = keras.layers.Dense(
            num_classes,
            activation=activation,
            name="predictions",
            dtype="float32",
        )(x)
        super().__init__(inputs=inputs, outputs=outputs, name="VideoSwinS", **kwargs)
        self.window_size = window_size
        self.num_classes = num_classes
        self.pooling = pooling
        self.activation = activation
        self.embed_size = embed_size
        self.depths = depths
        self.num_heads = num_heads
        self.include_rescaling = include_rescaling
        self.include_top = include_top

    def get_config(self):
        config = {
            "input_shape": self.input_shape[1:],
            "window_size": self.window_size,
            "num_classes": self.num_classes,
            "pooling": self.pooling,
            "activation": self.activation,
            "embed_size": self.embed_size,
            "depths": self.depths,
            "num_heads": self.num_heads,
            "include_rescaling": self.include_rescaling,
            "include_top": self.include_top,
        }
        return config


@keras.utils.register_keras_serializable(package="swin.transformer.base.3d")
def VideoSwinB(
    input_shape=(32, 224, 224, 3),
    num_classes=400,
    pooling="avg",
    activation="softmax",
    window_size=(8, 7, 7),
    embed_size=128,
    depths=[2, 2, 18, 2],
    num_heads=[4, 8, 16, 32],
    include_rescaling=False,
    include_top=True,
    **kwargs,
):
    if pooling == "avg":
        pooling_layer = keras.layers.GlobalAveragePooling3D(name="avg_pool")
    elif pooling == "max":
        pooling_layer = keras.layers.GlobalMaxPooling3D(name="max_pool")
    else:
        raise ValueError(f'`pooling` must be one of "avg", "max". Received: {pooling}.')

    backbone = VideoSwinBackbone(
        input_shape=input_shape,
        window_size=window_size,
        embed_dim=embed_size,
        depths=depths,
        num_heads=num_heads,
        include_rescaling=include_rescaling,
        name="VideoSwinBBackbone",
    )

    if not include_top:
        return backbone

    inputs = backbone.input
    x = backbone(inputs)
    x = pooling_layer(x)
    outputs = keras.layers.Dense(
        num_classes,
        activation=activation,
        name="predictions",
        dtype="float32",
    )(x)

    return keras.Model(inputs=inputs, outputs=outputs, name="VideoSwinB", **kwargs)


# @keras.utils.register_keras_serializable(package="swin.transformer.base.3d")
# class VideoSwinB(keras.Model):
#     def __init__(
#         self,
#         input_shape=(32, 224, 224, 3),
#         num_classes=400,
#         pooling="avg",
#         activation="softmax",
#         window_size=(8, 7, 7),
#         embed_size=128,
#         depths=[2, 2, 18, 2],
#         num_heads=[4, 8, 16, 32],
#         include_rescaling=False,
#         include_top=True,
#         **kwargs,
#     ):

#         if pooling == "avg":
#             pooling_layer = keras.layers.GlobalAveragePooling3D(name="avg_pool")
#         elif pooling == "max":
#             pooling_layer = keras.layers.GlobalMaxPooling3D(name="max_pool")
#         else:
#             raise ValueError(
#                 f'`pooling` must be one of "avg", "max". Received: {pooling}.'
#             )

#         backbone = VideoSwinBackbone(
#             input_shape=input_shape,
#             embed_dim=embed_size,
#             window_size=window_size,
#             depths=depths,
#             num_heads=num_heads,
#             include_rescaling=include_rescaling,
#         )

#         if not include_top:
#             return backbone

#         inputs = backbone.input
#         x = backbone(inputs)
#         x = pooling_layer(x)
#         outputs = keras.layers.Dense(
#             num_classes,
#             activation=activation,
#             name="predictions",
#             dtype="float32",
#         )(x)
#         super().__init__(inputs=inputs, outputs=outputs, name="VideoSwinB", **kwargs)
#         self.window_size = window_size
#         self.num_classes = num_classes
#         self.pooling = pooling
#         self.activation = activation
#         self.embed_size = embed_size
#         self.depths = depths
#         self.num_heads = num_heads
#         self.include_rescaling = include_rescaling
#         self.include_top = include_top

#     def get_config(self):
#         config = {
#             "input_shape": self.input_shape[1:],
#             "window_size": self.window_size,
#             "num_classes": self.num_classes,
#             "pooling": self.pooling,
#             "activation": self.activation,
#             "embed_size": self.embed_size,
#             "depths": self.depths,
#             "num_heads": self.num_heads,
#             "include_rescaling": self.include_rescaling,
#             "include_top": self.include_top,
#         }
#         return config

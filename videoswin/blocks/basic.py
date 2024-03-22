import keras
import tensorflow as tf
from keras import ops

from ..utils import compute_mask, get_window_size
from .swin_transformer import VideoSwinTransformerBlock


class VideoSwinBasicLayer(keras.Model):
    """A basic Video Swin Transformer layer for one stage.

    Args:
        input_dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (tuple[int]): Local window size. Default: (1,7,7).
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (keras.layers, optional): Normalization layer. Default: LayerNormalization
        downsample (keras.layers | None, optional): Downsample layer at the end of the layer. Default: None

    References:
        - [Video Swin Transformer](https://arxiv.org/abs/2106.13230)
        - [Video Swin Transformer GitHub](https://github.com/SwinTransformer/Video-Swin-Transformer)
    """  # noqa: E501

    def __init__(
        self,
        input_dim,
        depth,
        num_heads,
        window_size=(1, 7, 7),
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=None,
        downsample=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.shift_size = tuple([i // 2 for i in window_size])
        self.depth = depth
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.norm_layer = norm_layer
        self.downsample = downsample

    def _compute_dim_padded(self, input_dim, window_dim_size):
        input_dim = ops.cast(input_dim, dtype="float32")
        window_dim_size = ops.cast(window_dim_size, dtype="float32")
        return ops.cast(
            ops.ceil(input_dim / window_dim_size) * window_dim_size, "int32"
        )

    def build(self, input_shape):
        self.window_size, self.shift_size = get_window_size(
            input_shape[1:-1], self.window_size, self.shift_size
        )
        self.depth_pad = self._compute_dim_padded(input_shape[1], self.window_size[0])
        self.height_pad = self._compute_dim_padded(input_shape[2], self.window_size[1])
        self.width_pad = self._compute_dim_padded(input_shape[3], self.window_size[2])
        self.attn_mask = compute_mask(
            self.depth_pad,
            self.height_pad,
            self.width_pad,
            self.window_size,
            self.shift_size,
        )

        # build blocks
        self.blocks = [
            VideoSwinTransformerBlock(
                self.input_dim,
                num_heads=self.num_heads,
                window_size=self.window_size,
                shift_size=(0, 0, 0) if (i % 2 == 0) else self.shift_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                qk_scale=self.qk_scale,
                drop_rate=self.drop_rate,
                attn_drop_rate=self.attn_drop_rate,
                drop_path_rate=(
                    self.drop_path_rate[i]
                    if isinstance(self.drop_path_rate, list)
                    else self.drop_path_rate
                ),
                norm_layer=self.norm_layer,
            )
            for i in range(self.depth)
        ]

        if self.downsample is not None:
            self.downsample = self.downsample(
                input_dim=self.input_dim, norm_layer=self.norm_layer
            )
            self.downsample.build(input_shape)

        for i in range(self.depth):
            self.blocks[i].build(input_shape)

        self.built = True

    def call(self, x, training=None):
        input_shape = ops.shape(x)
        batch_size, depth, height, width, channel = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
            input_shape[4],
        )

        for block in self.blocks:
            x = block(x, self.attn_mask, training=training)

        x = ops.reshape(x, [batch_size, depth, height, width, channel])

        if self.downsample is not None:
            x = self.downsample(x)

        return x
    
    # def compute_output_shape(self, input_shape):
    #     # if self.downsample is not None:
    #     #     # TODO: remove tensorflow dependencies.
    #     #     # GitHub issue: https://github.com/keras-team/keras/issues/19259 # noqa: E501
    #     #     output_shape = tf.TensorShape(
    #     #         [
    #     #             input_shape[0],
    #     #             self.depth_pad,
    #     #             self.height_pad // 2,
    #     #             self.width_pad // 2,
    #     #             2 * self.input_dim,
    #     #         ]
    #     #     )
    #     #     return output_shape

    #     return input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "input_dim": self.input_dim,
                "window_size": self.window_size,
                "num_heads": self.num_heads,
                "mlp_ratio": self.mlp_ratio,
                "shift_size": self.shift_size,
                "depth": self.depth,
                "qkv_bias": self.qkv_bias,
                "qk_scale": self.qk_scale,
                "drop_rate": self.drop_rate,
                "attn_drop_rate": self.attn_drop_rate,
                "drop_path_rate": self.drop_path_rate,
            }
        )
        return config

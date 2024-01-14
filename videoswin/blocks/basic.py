from functools import partial
from typing import Optional, Tuple, Type

import keras
from keras import layers, ops

from ..utils import compute_mask, get_window_size
from .swin_transformer import SwinTransformerBlock3D


class BasicLayer(keras.Model):
    """A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of feature channels
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
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: Tuple[int, int, int] = (1, 7, 7),
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        norm_layer: Type[layers.Layer] = partial(
            layers.LayerNormalization, epsilon=1e-05
        ),
        downsample: Optional[Type[layers.Layer]] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.shift_size = tuple([i // 2 for i in window_size])
        self.depth = depth
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.drop_rate = drop_rate
        self.attn_drop = attn_drop
        self.drop_path = drop_path
        self.norm_layer = norm_layer
        self.downsample = downsample

    def compute_dim_padded(self, input_dim, window_dim_size):
        input_dim = ops.cast(input_dim, dtype="float32")
        window_dim_size = ops.cast(window_dim_size, dtype="float32")
        return ops.cast(
            ops.ceil(input_dim / window_dim_size) * window_dim_size, "int32"
        )

    def compute_output_shape(self, input_shape):
        window_size, _ = get_window_size(
            input_shape[1:-1], self.window_size, self.shift_size
        )
        depth_p = self.compute_dim_padded(input_shape[1], window_size[0])
        height_p = self.compute_dim_padded(input_shape[2], window_size[1])
        width_p = self.compute_dim_padded(input_shape[3], window_size[2])
        output_shape = (input_shape[0], depth_p, height_p, width_p, self.dim)
        return output_shape

    def build(self, input_shape):
        window_size, shift_size = get_window_size(
            input_shape[1:-1], self.window_size, self.shift_size
        )
        depth_p = self.compute_dim_padded(input_shape[1], window_size[0])
        height_p = self.compute_dim_padded(input_shape[2], window_size[1])
        width_p = self.compute_dim_padded(input_shape[3], window_size[2])
        self.attn_mask = compute_mask(
            depth_p, height_p, width_p, window_size, shift_size
        )

        # build blocks
        self.blocks = [
            SwinTransformerBlock3D(
                dim=self.dim,
                num_heads=self.num_heads,
                window_size=self.window_size,
                shift_size=(0, 0, 0) if (i % 2 == 0) else self.shift_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                qk_scale=self.qk_scale,
                drop_rate=self.drop_rate,
                attn_drop=self.attn_drop,
                drop_path=self.drop_path[i]
                if isinstance(self.drop_path, list)
                else self.drop_path,
                norm_layer=self.norm_layer,
            )
            for i in range(self.depth)
        ]

        if self.downsample is not None:
            self.downsample = self.downsample(dim=self.dim, norm_layer=self.norm_layer)

    def call(self, x, training=None, return_attention_maps=False):
        input_shape = ops.shape(x)
        batch_size, depth, height, width, channel = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
            input_shape[4],
        )

        for block in self.blocks:
            if return_attention_maps:
                x, attention_maps = block(
                    x,
                    self.attn_mask,
                    return_attention_maps=return_attention_maps,
                    training=training,
                )
            else:
                x = block(x, self.attn_mask, training=training)

        x = ops.reshape(x, [batch_size, depth, height, width, -1])

        if self.downsample is not None:
            x = self.downsample(x)

        if return_attention_maps:
            return x, attention_maps

        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "window_size": self.window_size,
                "num_heads": self.num_heads,
                "mlp_ratio": self.mlp_ratio,
                "shift_size": self.shift_size,
                "depth": self.depth,
                "qkv_bias": self.qkv_bias,
                "qk_scale": self.qk_scale,
                "drop": self.drop,
                "attn_drop": self.attn_drop,
                "drop_path": self.drop_path,
            }
        )
        return config

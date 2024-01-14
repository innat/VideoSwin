from typing import Optional, Tuple

import keras
from keras import layers, ops


class WindowAttention3D(keras.Model):
    """Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The temporal length, height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(
        self,
        dim: int,
        window_size: Tuple[int, int, int],
        num_heads: int,
        qkv_bias: Optional[bool] = True,
        qk_scale: Optional[float] = None,
        attn_drop: Optional[float] = 0.0,
        proj_drop: Optional[float] = 0.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        # variables
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv_bias = qkv_bias
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop

    def get_relative_position_index(self, window_depth, window_height, window_width):
        y_y, z_z, x_x = ops.meshgrid(
            range(window_width), range(window_depth), range(window_height)
        )
        coords = ops.stack([z_z, y_y, x_x], axis=0)
        coords_flatten = ops.reshape(coords, [3, -1])
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = ops.transpose(relative_coords, [1, 2, 0])
        z_z = (
            (relative_coords[:, :, 0] + window_depth - 1)
            * (2 * window_height - 1)
            * (2 * window_width - 1)
        )
        x_x = (relative_coords[:, :, 1] + window_height - 1) * (2 * window_width - 1)
        y_y = relative_coords[:, :, 2] + window_width - 1
        relative_coords = ops.stack([z_z, x_x, y_y], axis=-1)
        return ops.sum(relative_coords, axis=-1)

    def build(self, input_shape):
        self.relative_position_bias_table = self.add_weight(
            shape=(
                (2 * self.window_size[0] - 1)
                * (2 * self.window_size[1] - 1)
                * (2 * self.window_size[2] - 1),
                self.num_heads,
            ),
            initializer="zeros",
            trainable=True,
            name="relative_position_bias_table",
        )
        self.relative_position_index = self.get_relative_position_index(
            self.window_size[0], self.window_size[1], self.window_size[2]
        )

        # layers
        self.qkv = layers.Dense(self.dim * 3, use_bias=self.qkv_bias)
        self.attn_drop = layers.Dropout(self.attn_drop)
        self.proj = layers.Dense(self.dim)
        self.proj_drop = layers.Dropout(self.proj_drop)

    def call(self, x, mask=None, return_attention_maps=False, training=None):
        input_shape = ops.shape(x)
        batch_size, depth, channel = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
        )

        qkv = self.qkv(x)
        qkv = ops.reshape(
            qkv, [batch_size, depth, 3, self.num_heads, channel // self.num_heads]
        )
        qkv = ops.transpose(qkv, [2, 0, 3, 1, 4])
        q, k, v = ops.split(qkv, 3, axis=0)

        q = ops.squeeze(q, axis=0) * self.scale
        k = ops.squeeze(k, axis=0)
        v = ops.squeeze(v, axis=0)
        attention_maps = ops.matmul(q, ops.transpose(k, [0, 1, 3, 2]))

        relative_position_bias = ops.take(
            self.relative_position_bias_table,
            self.relative_position_index[:depth, :depth],
        )
        relative_position_bias = ops.reshape(relative_position_bias, [depth, depth, -1])
        relative_position_bias = ops.transpose(relative_position_bias, [2, 0, 1])
        attention_maps = attention_maps + relative_position_bias[None, ...]

        if mask is not None:
            mask_size = ops.shape(mask)[0]
            mask = ops.cast(mask, dtype=attention_maps.dtype)
            attention_maps = ops.reshape(
                attention_maps,
                [batch_size // mask_size, mask_size, self.num_heads, depth, depth],
            )
            attention_maps = attention_maps + mask[:, None, :, :]
            attention_maps = ops.reshape(
                attention_maps, [-1, self.num_heads, depth, depth]
            )

        attention_maps = keras.activations.softmax(attention_maps, axis=-1)
        attention_maps = self.attn_drop(attention_maps, training=training)

        x = ops.matmul(attention_maps, v)
        x = ops.transpose(x, [0, 2, 1, 3])
        x = ops.reshape(x, [batch_size, depth, channel])
        x = self.proj(x)
        x = self.proj_drop(x, training=training)

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
                "scale": self.scale,
                "qkv_bias": self.qkv_bias,
                "attn_drop": self.attn_drop,
                "proj_drop": self.proj_drop,
            }
        )
        return config

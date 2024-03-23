import keras
from keras import layers, ops


class VideoSwinWindowAttention(keras.Model):
    """It tackles long-range video dependencies by splitting features into windows
    and using relative position bias within each window for focused attention.
    It supports both of shifted and non-shifted window.

    Args:
        input_dim (int): The number of input channels in the feature maps.
        window_size (tuple[int]): The temporal length, height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop_rate (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop_rate (float, optional): Dropout ratio of output. Default: 0.0

    References:
        - [Video Swin Transformer](https://arxiv.org/abs/2106.13230)
        - [Video Swin Transformer GitHub](https://github.com/SwinTransformer/Video-Swin-Transformer)
    """  # noqa: E501

    def __init__(
        self,
        input_dim,
        window_size,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop_rate=0.0,
        proj_drop_rate=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # variables
        self.input_dim = input_dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = input_dim // num_heads
        self.qk_scale = qk_scale
        self.scale = qk_scale or head_dim**-0.5
        self.qkv_bias = qkv_bias
        self.attn_drop_rate = attn_drop_rate
        self.proj_drop_rate = proj_drop_rate

    def get_relative_position_index(self, window_depth, window_height, window_width):
        y_y, z_z, x_x = ops.meshgrid(
            ops.arange(window_width),
            ops.arange(window_depth),
            ops.arange(window_height),
        )
        coords = ops.stack([z_z, y_y, x_x], axis=0)
        coords_flatten = ops.reshape(coords, [3, -1])
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = ops.transpose(relative_coords, axes=[1, 2, 0])
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
        self.qkv = layers.Dense(self.input_dim * 3, use_bias=self.qkv_bias)
        self.attn_drop = layers.Dropout(self.attn_drop_rate)
        self.proj = layers.Dense(self.input_dim)
        self.proj_drop = layers.Dropout(self.proj_drop_rate)
        self.qkv.build(input_shape)
        self.proj.build(input_shape)
        self.built = True

    def call(self, x, mask=None, training=None):
        input_shape = ops.shape(x)
        batch_size, depth, channel = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
        )

        qkv = self.qkv(x)
        qkv = ops.reshape(
            qkv,
            [batch_size, depth, 3, self.num_heads, channel // self.num_heads],
        )
        qkv = ops.transpose(qkv, [2, 0, 3, 1, 4])
        q, k, v = ops.split(qkv, 3, axis=0)
        q = ops.squeeze(q, axis=0) * self.scale
        k = ops.squeeze(k, axis=0)
        v = ops.squeeze(v, axis=0)
        attn = ops.matmul(q, ops.transpose(k, [0, 1, 3, 2]))

        rel_pos_bias = ops.take(
            self.relative_position_bias_table,
            self.relative_position_index[:depth, :depth],
            axis=0,
        )
        rel_pos_bias = ops.reshape(rel_pos_bias, [depth, depth, -1])
        rel_pos_bias = ops.transpose(rel_pos_bias, [2, 0, 1])
        attn = attn + rel_pos_bias[None, ...]

        if mask is not None:
            mask_size = ops.shape(mask)[0]
            mask = ops.cast(mask, dtype=attn.dtype)
            attn = (
                ops.reshape(
                    attn,
                    [
                        batch_size // mask_size,
                        mask_size,
                        self.num_heads,
                        depth,
                        depth,
                    ],
                )
                + mask[:, None, :, :]
            )
            attn = ops.reshape(attn, [-1, self.num_heads, depth, depth])

        attn = keras.activations.softmax(attn, axis=-1)
        attn = self.attn_drop(attn, training=training)
        x = ops.matmul(attn, v)
        x = ops.transpose(x, [0, 2, 1, 3])
        x = ops.reshape(x, [batch_size, depth, channel])
        x = self.proj(x)
        x = self.proj_drop(x, training=training)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "input_dim": self.input_dim,
                "window_size": self.window_size,
                "num_heads": self.num_heads,
                "qk_scale": self.qk_scale,
                "qkv_bias": self.qkv_bias,
                "attn_drop_rate": self.attn_drop_rate,
                "proj_drop_rate": self.proj_drop_rate,
            }
        )
        return config

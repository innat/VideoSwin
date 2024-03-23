import keras
from keras import layers, ops

from ..layers import MLP, DropPath, VideoSwinWindowAttention
from ..utils import get_window_size, window_partition, window_reverse


class VideoSwinTransformerBlock(keras.Model):
    """Video Swin Transformer Block.

    Args:
        input_dim (int): Number of feature channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Local window size. Default: (2, 7, 7)
        shift_size (tuple[int]): Shift size for SW-MSA. Default: (0, 0, 0)
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            Default: 4.0
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value.
            Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
            Default: None
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optionalc): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (keras.layers.Activation, optional): Activation layer. Default: gelu
        norm_layer (keras.layers, optional): Normalization layer.
            Default: LayerNormalization

    References:
        - [Video Swin Transformer](https://arxiv.org/abs/2106.13230)
        - [Video Swin Transformer GitHub](https://github.com/SwinTransformer/Video-Swin-Transformer)
    """  # noqa: E501

    def __init__(
        self,
        input_dim,
        num_heads,
        window_size=(2, 7, 7),
        shift_size=(0, 0, 0),
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        activation="gelu",
        norm_layer=layers.LayerNormalization,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # variables
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.mlp_hidden_dim = int(input_dim * mlp_ratio)
        self.norm_layer = norm_layer
        self._activation_identifier = activation

        for i, (shift, window) in enumerate(zip(self.shift_size, self.window_size)):
            if not (0 <= shift < window):
                raise ValueError(
                    f"shift_size[{i}] must be in the range 0 to less than "
                    f"window_size[{i}], but got shift_size[{i}]={shift} "
                    f"and window_size[{i}]={window}."
                )

    def build(self, input_shape):
        self.window_size, self.shift_size = get_window_size(
            input_shape[1:-1], self.window_size, self.shift_size
        )

        self.apply_cyclic_shift = False
        if any(i > 0 for i in self.shift_size):
            self.apply_cyclic_shift = True

        # layers
        self.drop_path = (
            DropPath(self.drop_path_rate)
            if self.drop_path_rate > 0.0
            else layers.Identity()
        )

        self.norm1 = self.norm_layer(axis=-1, epsilon=1e-05)
        self.norm1.build(input_shape)

        self.attn = VideoSwinWindowAttention(
            self.input_dim,
            window_size=self.window_size,
            num_heads=self.num_heads,
            qkv_bias=self.qkv_bias,
            qk_scale=self.qk_scale,
            attn_drop_rate=self.attn_drop_rate,
            proj_drop_rate=self.drop_rate,
        )
        self.attn.build((None, None, self.input_dim))

        self.norm2 = self.norm_layer(axis=-1, epsilon=1e-05)
        self.norm2.build((*input_shape[:-1], self.input_dim))

        self.mlp = MLP(
            output_dim=self.input_dim,
            hidden_dim=self.mlp_hidden_dim,
            activation=self._activation_identifier,
            drop_rate=self.drop_rate,
        )
        self.mlp.build((*input_shape[:-1], self.input_dim))

        # compute padding if needed.
        # pad input feature maps to multiples of window size.
        _, depth, height, width, _ = input_shape
        pad_l = pad_t = pad_d0 = 0
        self.pad_d1 = ops.mod(-depth + self.window_size[0], self.window_size[0])
        self.pad_b = ops.mod(-height + self.window_size[1], self.window_size[1])
        self.pad_r = ops.mod(-width + self.window_size[2], self.window_size[2])
        self.pads = [
            [0, 0],
            [pad_d0, self.pad_d1],
            [pad_t, self.pad_b],
            [pad_l, self.pad_r],
            [0, 0],
        ]
        self.built = True

    def first_forward(self, x, mask_matrix, training):
        input_shape = ops.shape(x)
        batch_size, depth, height, width, _ = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
            input_shape[4],
        )
        x = self.norm1(x)

        # apply padding if needed.
        x = ops.pad(x, self.pads)

        input_shape = ops.shape(x)
        depth_pad, height_pad, width_pad = (
            input_shape[1],
            input_shape[2],
            input_shape[3],
        )

        # cyclic shift
        if self.apply_cyclic_shift:
            shifted_x = ops.roll(
                x,
                shift=(
                    -self.shift_size[0],
                    -self.shift_size[1],
                    -self.shift_size[2],
                ),
                axis=(1, 2, 3),
            )
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)

        # get attentions params
        attn_windows = self.attn(x_windows, mask=attn_mask, training=training)

        # reverse the swin windows
        shifted_x = window_reverse(
            attn_windows,
            self.window_size,
            batch_size,
            depth_pad,
            height_pad,
            width_pad,
        )

        # reverse cyclic shift
        if self.apply_cyclic_shift:
            x = ops.roll(
                shifted_x,
                shift=(
                    self.shift_size[0],
                    self.shift_size[1],
                    self.shift_size[2],
                ),
                axis=(1, 2, 3),
            )
        else:
            x = shifted_x

        # pad if required
        do_pad = ops.logical_or(
            ops.greater(self.pad_d1, 0),
            ops.logical_or(ops.greater(self.pad_r, 0), ops.greater(self.pad_b, 0)),
        )
        x = ops.cond(do_pad, lambda: x[:, :depth, :height, :width, :], lambda: x)

        return x

    def second_forward(self, x, training):
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x, training=training)
        return x

    def call(self, x, mask_matrix=None, training=None):
        shortcut = x
        x = self.first_forward(x, mask_matrix, training)
        x = shortcut + self.drop_path(x)
        x = x + self.second_forward(x, training)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "input_dim": self.input_dim,
                "window_size": self.num_heads,
                "num_heads": self.window_size,
                "shift_size": self.shift_size,
                "mlp_ratio": self.mlp_ratio,
                "qkv_bias": self.qkv_bias,
                "qk_scale": self.qk_scale,
                "drop_rate": self.drop_rate,
                "attn_drop_rate": self.attn_drop_rate,
                "drop_path_rate": self.drop_path_rate,
                "mlp_hidden_dim": self.mlp_hidden_dim,
                "activation": self._activation_identifier,
            }
        )
        return config

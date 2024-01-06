import keras
from keras import layers, ops

from ..layers import MLP, DropPath, WindowAttention3D
from ..utils import get_window_size, window_partition, window_reverse


class SwinTransformerBlock3D(keras.Model):
    """Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Window size.
        shift_size (tuple[int]): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (keras.layers.Activation, optional): Activation layer. Default: gelu
        norm_layer (keras.layers, optional): Normalization layer.  Default: LayerNormalization
    """

    def __init__(
        self,
        dim,
        num_heads,
        window_size=(2, 7, 7),
        shift_size=(0, 0, 0),
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=layers.Activation("gelu"),
        norm_layer=layers.LayerNormalization,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.attn_drop = attn_drop
        self.drop_rate = drop_rate
        self.drop_path = drop_path
        self.mlp_hidden_dim = int(dim * mlp_ratio)
        self.act_layer = act_layer
        self.norm_layer = norm_layer

        assert (
            0 <= self.shift_size[0] < self.window_size[0]
        ), "shift_size must in 0-window_size"
        assert (
            0 <= self.shift_size[1] < self.window_size[1]
        ), "shift_size must in 0-window_size"
        assert (
            0 <= self.shift_size[2] < self.window_size[2]
        ), "shift_size must in 0-window_size"

    def build(self, input_shape):
        self.window_size, self.shift_size = get_window_size(
            input_shape[1:-1], self.window_size, self.shift_size
        )
        if any(i > 0 for i in self.shift_size):
            self.roll = True
        else:
            self.roll = False

        # layers
        self.norm1 = self.norm_layer(axis=-1, epsilon=1e-05)
        self.attn = WindowAttention3D(
            self.dim,
            window_size=self.window_size,
            num_heads=self.num_heads,
            qkv_bias=self.qkv_bias,
            qk_scale=self.qk_scale,
            attn_drop=self.attn_drop,
            proj_drop=self.drop_rate,
        )
        self.drop_path = (
            DropPath(self.drop_path) if self.drop_path > 0.0 else layers.Identity()
        )
        self.norm2 = self.norm_layer(axis=-1, epsilon=1e-05)
        self.mlp = MLP(
            in_features=self.dim,
            hidden_features=self.mlp_hidden_dim,
            act_layer=self.act_layer,
            drop_rate=self.drop_rate,
        )

    def _forward(self, x, mask_matrix, return_attention_maps, training):
        input_shape = ops.shape(x)
        batch_size, depth, height, width, channel = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
            input_shape[4],
        )
        window_size, shift_size = self.window_size, self.shift_size
        x = self.norm1(x)

        # pad feature maps to multiples of window size
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = ops.mod(-depth + window_size[0], window_size[0])
        pad_b = ops.mod(-height + window_size[1], window_size[1])
        pad_r = ops.mod(-width + window_size[2], window_size[2])
        paddings = [[0, 0], [pad_d0, pad_d1], [pad_t, pad_b], [pad_l, pad_r], [0, 0]]
        x = ops.pad(x, paddings)

        input_shape = ops.shape(x)
        depth_p, height_p, width_p = (
            input_shape[1],
            input_shape[2],
            input_shape[3],
        )

        # Cyclic Shift
        if self.roll:
            shifted_x = ops.roll(
                x,
                shift=(-shift_size[0], -shift_size[1], -shift_size[2]),
                axis=(1, 2, 3),
            )
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, window_size)

        # get attentions params
        if return_attention_maps:
            attention_windows, attention_maps = self.attn(
                x_windows,
                mask=attn_mask,
                return_attention_maps=return_attention_maps,
                training=training,
            )
        else:
            attention_windows = self.attn(x_windows, mask=attn_mask, training=training)

        # reverse the swin windows
        shifted_x = window_reverse(
            attention_windows, window_size, batch_size, depth_p, height_p, width_p
        )

        # Reverse Cyclic Shift
        if self.roll:
            x = ops.roll(
                shifted_x,
                shift=(shift_size[0], shift_size[1], shift_size[2]),
                axis=(1, 2, 3),
            )
        else:
            x = shifted_x

        # pad if required
        do_pad = ops.logical_or(
            ops.greater(pad_d1, 0),
            ops.logical_or(ops.greater(pad_r, 0), ops.greater(pad_b, 0)),
        )
        x = ops.cond(do_pad, lambda: x[:, :depth, :height, :width, :], lambda: x)

        if return_attention_maps:
            return x, attention_maps

        return x

    def call(self, x, mask_matrix=None, return_attention_maps=False, training=None):
        shortcut = x
        x = self._forward(x, mask_matrix, return_attention_maps, training)

        if return_attention_maps:
            x, attention_maps = x

        x = shortcut + self.drop_path(x)
        x = self.drop_path(self.mlp(self.norm2(x)), training=training)

        if return_attention_maps:
            return x, attention_maps

        return x

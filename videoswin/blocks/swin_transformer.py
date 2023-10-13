
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from ..layers import TFMlp
from ..layers import TFWindowAttention3D
from ..layers import TFDropPath
from ..utils import get_window_size
from ..utils import tf_window_partition
from ..utils import tf_window_reverse

class TFSwinTransformerBlock3D(keras.Model):
    """ Swin Transformer Block.

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
        mlp_ratio=4., 
        qkv_bias=True, 
        qk_scale=None, 
        drop=0., 
        attn_drop=0., 
        drop_path=0.,
        act_layer=layers.Activation('gelu'), 
        norm_layer=layers.LayerNormalization, 
        **kwargs
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        mlp_hidden_dim = int(dim * mlp_ratio)
        
        assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[2] < self.window_size[2], "shift_size must in 0-window_size"
        
        # layers
        self.norm1 = norm_layer(axis=-1, epsilon=1e-05)
        self.attn = TFWindowAttention3D(
            dim, 
            window_size=window_size, 
            num_heads=num_heads, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            attn_drop=attn_drop,
            proj_drop=drop
        )
        self.drop_path = TFDropPath(drop_path) if drop_path > 0. else layers.Identity()  
        self.norm2 = norm_layer(axis=-1, epsilon=1e-05)
        self.mlp = TFMlp(
            in_features=dim, 
            hidden_features=mlp_hidden_dim, 
            act_layer=act_layer, 
            drop=drop
        )
        
    def build(self, input_shape):
        self.window_size, self.shift_size = get_window_size(
            input_shape[1:-1], self.window_size, self.shift_size
        )
        if any(i > 0 for i in self.shift_size):
            self.roll = True
        else:
            self.roll = False
        
        super().build(input_shape)
        

    def first_forward(self, x, mask_matrix, return_attns, training):
        input_shape = tf.shape(x)
        B,D,H,W,C = (
            input_shape[0], 
            input_shape[1],
            input_shape[2],
            input_shape[3],
            input_shape[4],
        )
        window_size, shift_size = self.window_size, self.shift_size
        x = self.norm1(x)
        
        # pad feature maps to multiples of window size
        pad_l  = pad_t = pad_d0 = 0
        pad_d1 = tf.math.floormod(-D + window_size[0], window_size[0])
        pad_b  = tf.math.floormod(-H + window_size[1], window_size[1])
        pad_r  = tf.math.floormod(-W + window_size[2], window_size[2])
        paddings = [[0, 0], [pad_d0, pad_d1], [pad_t, pad_b], [pad_l, pad_r], [0, 0]]
        x = tf.pad(x, paddings)
        
        input_shape = tf.shape(x)
        Dp, Hp, Wp =  (
            input_shape[1],
            input_shape[2],
            input_shape[3],
        )
        
        # Cyclic Shift
        if self.roll:
            shifted_x = tf.roll(
                x, 
                shift=(-shift_size[0], -shift_size[1], -shift_size[2]), 
                axis=(1, 2, 3)
            )
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        
        # partition windows
        x_windows = tf_window_partition(shifted_x, window_size) 
        
        # get attentions params
        if return_attns:
            attn_windows, attn_scores = self.attn(
                x_windows, mask=attn_mask, return_attns=return_attns, training=training
            )
        else:
             attn_windows = self.attn(
                x_windows, mask=attn_mask, training=training
            ) 

        # reverse the swin windows
        shifted_x = tf_window_reverse(
            attn_windows, window_size, B, Dp, Hp, Wp
        ) 

        # Reverse Cyclic Shift
        if self.roll:
            x = tf.roll(
                shifted_x, 
                shift=(shift_size[0], shift_size[1], shift_size[2]), 
                axis=(1, 2, 3)
            )
        else:
            x = shifted_x

        # pad if required    
        do_pad = tf.logical_or(
            tf.greater(pad_d1, 0),
            tf.logical_or(tf.greater(pad_r, 0), tf.greater(pad_b, 0))
        )
        x = tf.cond(
            do_pad, 
            lambda: x[:, :D, :H, :W, :], 
            lambda: x
        )

        if return_attns:
            return x, attn_scores
        
        return x

    def second_forward(self, x, training):
        return self.drop_path(
            self.mlp(self.norm2(x)), training=training
        )

    def call(self, x, mask_matrix=None, return_attns=False, training=None):
        
        shortcut = x
        x = self.first_forward(
            x, mask_matrix, return_attns, training
        )
        
        if return_attns:
            x, attn_scores = x

        x = shortcut + self.drop_path(x)
        x = x + self.second_forward(x, training)
        
        if return_attns:
            return x, attn_scores
        
        return x
    
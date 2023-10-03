
from functools import partial

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import TFSwinTransformerBlock3D
from videoswin.utils import get_window_size
from videoswin.utils import tf_compute_mask

class TFBasicLayer(keras.Model):
    """ A basic Swin Transformer layer for one stage.

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
        dim,
        depth,
        num_heads,
        window_size=(1,7,7),
        mlp_ratio=4.,
        qkv_bias=False,
        qk_scale=None,
        drop=0.,
        attn_drop=0.,
        drop_path=0.,
        norm_layer=partial(layers.LayerNormalization, epsilon=1e-05),
        downsample=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.window_size = window_size
        self.shift_size = tuple([i // 2 for i in window_size])
        self.depth = depth

        # build blocks
        self.blocks = [
            TFSwinTransformerBlock3D(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0,0,0) if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
            )
            for i in range(depth)]

        self.downsample = downsample
        if self.downsample is not None:
            self.downsample = downsample(
                dim=dim, norm_layer=norm_layer
            )
    
    def compute_dim_padded(self, input_dim, window_dim_size):
        input_dim = tf.cast(input_dim, dtype=tf.float32)
        window_dim_size = tf.cast(window_dim_size, dtype=tf.float32)
        return tf.cast(
            tf.math.ceil(input_dim / window_dim_size) * window_dim_size,
            tf.int32
        )
    
    def build(self, input_shape):
        window_size, shift_size = get_window_size(
            input_shape[1:-1], self.window_size, self.shift_size
        )
        Dp = self.compute_dim_padded(input_shape[1], window_size[0])
        Hp = self.compute_dim_padded(input_shape[2], window_size[1])
        Wp = self.compute_dim_padded(input_shape[3], window_size[2])
        self.attn_mask = tf_compute_mask(
            Dp, Hp, Wp, window_size, shift_size
        )
        super().build(input_shape)
        

    def call(self, x, training=None, return_attns=False):
        input_shape = tf.shape(x)
        B,D,H,W,C = (
            input_shape[0], 
            input_shape[1],
            input_shape[2],
            input_shape[3],
            input_shape[4],
        )

        for blk in self.blocks:
            if return_attns:
                x, attn_scores = blk(
                    x, 
                    self.attn_mask,
                    return_attns=return_attns,
                    training=training
                )
            else:
                x = blk(
                    x, 
                    self.attn_mask,
                    training=training
                )
            
        x = tf.reshape(
            x, [B, D, H, W, -1]
        )
 
        if self.downsample is not None:
            x = self.downsample(x)
            
        if return_attns:
            return x, attn_scores
            
        return x
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class TFWindowAttention3D(keras.Model):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
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
        dim, 
        window_size, 
        num_heads, 
        qkv_bias=True, 
        qk_scale=None, 
        attn_drop=0., 
        proj_drop=0.,
        **kwargs
    ):
        super().__init__(**kwargs)
        # variables
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # layers
        self.qkv = layers.Dense(dim * 3, use_bias=qkv_bias)
        self.attn_drop = layers.Dropout(attn_drop)
        self.proj = layers.Dense(dim)
        self.proj_drop = layers.Dropout(proj_drop)
        
    def get_relative_position_index(self, window_depth, window_height, window_width):
        y_y, z_z, x_x = tf.meshgrid(
            range(window_width), range(window_depth), range(window_height)
        )
        coords = tf.stack([z_z, y_y, x_x], axis=0)
        coords_flatten = tf.reshape(coords, [3, -1])
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = tf.transpose(relative_coords, perm=[1, 2, 0])
        z_z = (relative_coords[:, :, 0] + window_depth  - 1) * (2 * window_height - 1) * (2 * window_width - 1)
        x_x = (relative_coords[:, :, 1] + window_height - 1) * (2 * window_width - 1)
        y_y = (relative_coords[:, :, 2] + window_width  - 1)
        relative_coords = tf.stack([z_z, x_x, y_y], axis=-1)
        return tf.reduce_sum(relative_coords, axis=-1)

    def build(self, input_shape):
        self.relative_position_bias_table = self.add_weight(
            shape=(
                (2 * self.window_size[0] - 1) * 
                (2 * self.window_size[1] - 1) * 
                (2 * self.window_size[2] - 1),
                self.num_heads,
            ),
            initializer="zeros",
            trainable=True,
            name="relative_position_bias_table",
        )
        self.relative_position_index = self.get_relative_position_index(
            self.window_size[0], self.window_size[1], self.window_size[2]
        )
        super().build(input_shape)


    def call(self, x, mask=None, return_attns=False, training=None):
        input_shape = tf.shape(x)
        B_,N,C = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
        )
        
        qkv = self.qkv(x)
        qkv = tf.reshape(qkv, [B_, N, 3, self.num_heads, C // self.num_heads])
        qkv = tf.transpose(qkv, perm=[2, 0, 3, 1, 4])
        q, k, v = tf.split(qkv, 3, axis=0)

        q = tf.squeeze(q, axis=0) * self.scale
        k = tf.squeeze(k, axis=0)
        v = tf.squeeze(v, axis=0)
        attn = tf.linalg.matmul(q, k, transpose_b=True)
        
        relative_position_bias = tf.gather(
            self.relative_position_bias_table, self.relative_position_index[:N, :N]
        )
        relative_position_bias = tf.reshape(relative_position_bias, [N, N, -1])
        relative_position_bias = tf.transpose(relative_position_bias, perm=[2, 0, 1])
        attn = attn + relative_position_bias[None, ...]
  
        if mask is not None:
            nW = tf.shape(mask)[0]
            mask = tf.cast(mask, dtype=attn.dtype)
            attn = tf.reshape(attn, [B_ // nW, nW, self.num_heads, N, N]) + mask[:, None, :, :]
            attn = tf.reshape(attn, [-1, self.num_heads, N, N])

        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn, training=training)

        x = tf.linalg.matmul(attn, v)
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        x = tf.reshape(x, [B_, N, C])
        x = self.proj(x)
        x = self.proj_drop(x, training=training)
        
        if return_attns:
            return x, attn
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "window_size": self.window_size,
                "num_heads": self.num_heads,
            }
        )
        return config
    
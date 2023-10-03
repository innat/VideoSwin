
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class TFPatchEmbed3D(keras.Model):
    """ Video to Patch Embedding.

    Args:
        patch_size (int): Patch token size. Default: (2,4,4).
        in_chans (int): Number of input video channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (keras.layers, optional): Normalization layer. Default: None
    """
        
    def __init__(
        self, 
        patch_size=(2, 4, 4), 
        in_chans=3, 
        embed_dim=96, 
        norm_layer=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        # layers
        self.proj = layers.Conv3D(
            embed_dim, 
            kernel_size=patch_size,
            strides=patch_size, 
            name='embed_proj'
        )
        if norm_layer is not None:
            self.norm = norm_layer(
                axis=-1, epsilon=1e-5, name='embed_norm'
            )
        else:
            self.norm = None
        
    def build(self, input_shape):
        self.pads = [
            [0, 0], 
            self._compute_padding(input_shape[1], self.patch_size[0]),
            self._compute_padding(input_shape[2], self.patch_size[1]),
            self._compute_padding(input_shape[3], self.patch_size[2]),
            [0, 0]   
        ]
        super().build(input_shape)

    def _compute_padding(self, dim, patch_size):
        pad_amount = patch_size - (dim % patch_size)
        return [
            0, pad_amount if pad_amount != patch_size else 0
        ]
        
    def call(self, x):
        
        x = tf.pad(x, self.pads)
        x = self.proj(x)
        
        if self.norm is not None:
            x = self.norm(x) 
            
        return x
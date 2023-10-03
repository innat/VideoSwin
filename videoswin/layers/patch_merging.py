
import tensorflow as tf
from tensorflow.keras import layers

class TFPatchMerging(layers.Layer):
    """ Patch Merging Layer

    Args:
        dim (int): Number of input channels.
        norm_layer (keras.layers, optional): Normalization layer.  Default: LayerNormalization
    """
        
    def __init__(
        self, 
        dim, 
        norm_layer=layers.LayerNormalization,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.reduction = layers.Dense(2 * dim, use_bias=False)
        self.norm = norm_layer(axis=-1, epsilon=1e-5)
        
    def call(self, x):
        """ call function.

        Args:
            x: Input feature, tensor size (B, D, H, W, C).
        """
        input_shape = tf.shape(x)
        H,W = (
            input_shape[2],
            input_shape[3],
        )

        # padding if needed
        paddings = [
            [0, 0], 
            [0, 0], 
            [0, tf.math.floormod(H, 2)], 
            [0, tf.math.floormod(W, 2)], 
            [0, 0]
        ]
        x = tf.pad(x, paddings)

        x0 = x[:, :, 0::2, 0::2, :]  # B D H/2 W/2 C
        x1 = x[:, :, 1::2, 0::2, :]  # B D H/2 W/2 C
        x2 = x[:, :, 0::2, 1::2, :]  # B D H/2 W/2 C
        x3 = x[:, :, 1::2, 1::2, :]  # B D H/2 W/2 C
        x = tf.concat([x0, x1, x2, x3], axis=-1)  # B D H/2 W/2 4*C
        x = self.norm(x)
        x = self.reduction(x)
        return x
    
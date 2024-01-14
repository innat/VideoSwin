from keras import layers, ops


class PatchMerging(layers.Layer):
    """Patch Merging Layer

    Args:
        dim (int): Number of input channels.
        norm_layer (keras.layers, optional): Normalization layer.  Default: LayerNormalization
    """

    def __init__(self, dim, norm_layer=layers.LayerNormalization, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.reduction = layers.Dense(2 * dim, use_bias=False)
        self.norm = norm_layer(axis=-1, epsilon=1e-5)

    def call(self, x):
        """call function.

        Args:
            x: Input feature, tensor size (batch, depth, height, width, channel).
        """
        input_shape = ops.shape(x)
        height, width = (
            input_shape[2],
            input_shape[3],
        )

        # padding if needed
        paddings = [
            [0, 0],
            [0, 0],
            [0, ops.mod(height, 2)],
            [0, ops.mod(width, 2)],
            [0, 0],
        ]
        x = ops.pad(x, paddings)

        x0 = x[:, :, 0::2, 0::2, :]  # B D H/2 W/2 C
        x1 = x[:, :, 1::2, 0::2, :]  # B D H/2 W/2 C
        x2 = x[:, :, 0::2, 1::2, :]  # B D H/2 W/2 C
        x3 = x[:, :, 1::2, 1::2, :]  # B D H/2 W/2 C
        x = ops.concatenate([x0, x1, x2, x3], axis=-1)  # B D H/2 W/2 4*C
        x = self.norm(x)
        x = self.reduction(x)
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
            }
        )
        return config

from keras import layers, ops


class VideoSwinPatchMerging(layers.Layer):
    """Patch Merging Layer in Video Swin Transformer models.

    This layer performs a downsampling step by merging four neighboring patches
    from the previous layer into a single patch in the output. It achieves this
    by concatenation and linear projection.

    Args:
        input_dim (int): Number of input channels in the feature maps.
        norm_layer (keras.layers, optional): Normalization layer.
            Default: LayerNormalization

    References:
        - [Video Swin Transformer](https://arxiv.org/abs/2106.13230)
        - [Video Swin Transformer GitHub](https://github.com/SwinTransformer/Video-Swin-Transformer)
    """  # noqa: E501

    def __init__(self, input_dim, norm_layer=None, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.norm_layer = norm_layer

    def build(self, input_shape):
        batch_size, depth, height, width, channel = input_shape
        self.reduction = layers.Dense(2 * self.input_dim, use_bias=False)
        self.reduction.build((batch_size, depth, height // 2, width // 2, 4 * channel))

        if self.norm_layer is not None:
            self.norm = self.norm_layer(axis=-1, epsilon=1e-5)
            self.norm.build((batch_size, depth, height // 2, width // 2, 4 * channel))

        # compute padding if needed
        self.pads = [
            [0, 0],
            [0, 0],
            [0, ops.mod(height, 2)],
            [0, ops.mod(width, 2)],
            [0, 0],
        ]
        self.built = True

    def call(self, x):
        # padding if needed
        x = ops.pad(x, self.pads)
        x0 = x[:, :, 0::2, 0::2, :]  # B D H/2 W/2 C
        x1 = x[:, :, 1::2, 0::2, :]  # B D H/2 W/2 C
        x2 = x[:, :, 0::2, 1::2, :]  # B D H/2 W/2 C
        x3 = x[:, :, 1::2, 1::2, :]  # B D H/2 W/2 C
        x = ops.concatenate([x0, x1, x2, x3], axis=-1)  # B D H/2 W/2 4*C

        if self.norm_layer is not None:
            x = self.norm(x)

        x = self.reduction(x)
        return x

    def compute_output_shape(self, input_shape):
        batch_size, depth, height, width, _ = input_shape
        return (batch_size, depth, height // 2, width // 2, 2 * self.input_dim)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "input_dim": self.input_dim,
            }
        )
        return config

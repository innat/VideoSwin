
import math
import tensorflow as tf
from tensorflow.keras import layers

class TFAdaptiveAveragePooling3D(layers.Layer):
    def __init__(self, output_size, **kwargs):
        super().__init__(**kwargs)
        self.output_size = output_size
        
    def compute_pool_size(self, x_dim, dim):
        return math.ceil(x_dim / dim)

    def call(self, inputs):
        # input_shape: bs, depth, h, w, 3
        x = self.compute_pool_size(inputs.shape[1], self.output_size[0])
        y = self.compute_pool_size(inputs.shape[2], self.output_size[1])
        z = self.compute_pool_size(inputs.shape[3], self.output_size[2])
        
        # output_shape: [bs, 1, 1, 1, channel_dim]
        avg_pool = layers.AveragePooling3D(
            pool_size=[x, y, z], strides=[x, y, z], padding="valid"
        )(inputs)
        
        # output_shape: [bs, channel_dim]
        avg_pool = tf.squeeze(
            avg_pool, [1,2,3]
        )
        
        return avg_pool
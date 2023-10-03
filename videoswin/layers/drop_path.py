import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class TFDropPath(layers.Layer):
    def __init__(self, rate=0.5, seed=None, **kwargs):
        super().__init__(**kwargs)
        self.rate = rate
        self.seed = seed

    def call(self, x, training=None):
        if self.rate == 0.0 or not training:
            return x
        else:
            keep_prob = 1 - self.rate
            drop_map_shape = (tf.shape(x)[0],) + (1,) * (len(x.shape) - 1)
            drop_map = keras.backend.random_bernoulli(
                drop_map_shape, p=keep_prob, seed=self.seed, dtype=x.dtype
            )
            x = x / keep_prob
            x = x * drop_map
            return x

    def get_config(self):
        config = super().get_config()
        config.update({"rate": self.rate, "seed": self.seed})
        return config
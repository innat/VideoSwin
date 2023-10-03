
import tensorflow as tf
from tensorflow.keras import layers

class TFMlp(layers.Layer):
    """ Multilayer perceptron."""
    
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        drop=0.0,
        act_layer=layers.Activation(tf.nn.gelu),
        **kwargs
    ):
        super().__init__(**kwargs)
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = layers.Dense(hidden_features)
        self.fc2 = layers.Dense(out_features)
        self.act = act_layer
        self.drop = layers.Dropout(drop)

    def call(self, x, training=None):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x, training=training)
        x = self.fc2(x)
        x = self.drop(x, training=training)
        return x
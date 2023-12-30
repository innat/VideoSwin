
from keras import layers

class MLP(layers.Layer):
    """ Multilayer perceptron."""
    
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        drop_rate=0.0,
        act_layer=layers.Activation("gelu"),
        **kwargs
    ):
        super().__init__(**kwargs)
        self.out_features = out_features or in_features
        self.hidden_features = hidden_features or in_features
        self.drop_rate = drop_rate
        self.act = act_layer
        
    def build(self, input_shape):
        self.fc1 = layers.Dense(self.hidden_features)
        self.fc2 = layers.Dense(self.out_features)
        self.dropout = layers.Dropout(self.drop_rate)

    def call(self, x, training=None):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x, training=training)
        x = self.fc2(x)
        x = self.dropout(x, training=training)
        return x
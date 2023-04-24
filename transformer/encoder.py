import tensorflow as tf

from transformer.attention import MultiAttentionHead
from transformer.positional import PositionalEmbeddings
from transformer.feed_forward import FeedForward


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, config, name='EncoderLayer'):
        super(EncoderLayer, self).__init__(name=name)
        self.config = config
        self.layer_norm1 = tf.keras.layers.LayerNormalization(input_shape=(self.config['input_size'] - 1,))
        self.layer_norm2 = tf.keras.layers.LayerNormalization()
        self.attention = MultiAttentionHead(self.config)
        self.ff = FeedForward(self.config)

    def call(self, embeds):
        x, m = embeds
        x = self.layer_norm1(x)
        x = x + self.attention([x, m])
        x = self.layer_norm2(x)
        x = self.ff(x)
        return x


class Encoder(tf.keras.layers.Layer):
    def __init__(self, config, name='Encoder'):
        super(Encoder, self).__init__(name=name)
        self.config = config
        self.embeddings = PositionalEmbeddings(self.config)
        self.layers = [EncoderLayer(self.config) for _ in range(config['num_hidden_layers'])]

    def call(self, inputs):
        x, m = inputs
        x = self.embeddings(x)
        for layer in self.layers:
            x = layer([x, m])
        return x
    
    def get_config(self):
        config = super(Encoder, self).get_config()
        config.update({"config": self.config})
        return config

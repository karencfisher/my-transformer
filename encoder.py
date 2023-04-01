import tensorflow as tf

from attention import MultiAttentionHead
from positional import PositionalEmbeddings
from feed_forward import FeedForward


class Encoder(tf.keras.layers.Layer):
    def __init__(self, config, name="Encoder"):
        super(Encoder, self).__init__(name=name)
        self.layer_norm1 = tf.keras.layers.LayerNormalization()
        self.layer_norm2 = tf.keras.layers.LayerNormalization()
        self.attention = MultiAttentionHead(config)
        self.ff = FeedForward(config)

    def call(self, embeds):
        x = self.layer_norm1(embeds)
        x = x + self.attention(x)
        x = self.layer_norm2(x)
        x = self.ff(x)
        return x

import tensorflow as tf


class FeedForward(tf.keras.layers.Layer):
    def __init__(self, config, name='FeedForward'):
        super(FeedForward, self).__init__(name=name)
        self.dense1 = tf.keras.layers.Dense(config['intermediate_size'])
        self.dense2 = tf.keras.layers.Dense(config['hidden_size'])
        self.gelu = tf.keras.activations.gelu
        self.dropout = tf.keras.layers.Dropout(config['dropout_p'])

    def call(self, x):
        x = self.dense1(x)
        x = self.gelu(x)
        x = self.dense2(x)
        x = self.dropout(x)
        return x
    

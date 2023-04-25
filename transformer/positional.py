import tensorflow as tf


class PositionalEmbeddings(tf.keras.layers.Layer):
    def __init__(self, config, name='positinal_embedding'):
        super(PositionalEmbeddings, self).__init__(name=name)
        self.token_embeddings = tf.keras.layers.Embedding(config['vocab_size'],
                                                          config['hidden_size'],
                                                          mask_zero=True)
        self.pos_embeddings = tf.keras.layers.Embedding(config['max_position_embeds'],
                                                        config['hidden_size'],
                                                        mask_zero=True)
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-12)
        self.dropout = tf.keras.layers.Dropout(0.5)

    def call(self, input_ids):
        seq_length = input_ids.shape[1]
        position_ids = tf.range(seq_length, dtype=tf.int64)
        position_ids = tf.expand_dims(position_ids, 0)

        token_embeddings = self.token_embeddings(input_ids)
        position_embeddings = self.pos_embeddings(position_ids)

        embeddings = token_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


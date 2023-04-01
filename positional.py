import tensorflow as tf
import tiktoken


class PositionalEmbeddings(tf.keras.layers.Layer):
    def __init__(self, config, name='positinal_embedding'):
        super(PositionalEmbeddings, self).__init__(name=name)
        self.token_embeddings = tf.keras.layers.Embedding(config['vocab_size'],
                                                          config['hidden_size'])
        self.pos_embeddings = tf.keras.layers.Embedding(config['max_position_embeds'],
                                                        config['hidden_size'])
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


def test():
    config = {'num_heads': 4, 
              'vocab_size': 50257,
              'hidden_size': 8,
              'max_position_embeds': 50257}
    sentence = 'It is a good day to have lunch'
    encoder = tiktoken.get_encoding('p50k_base')
    sentence_enc = tf.convert_to_tensor(encoder.encode(sentence), dtype=tf.int64)
    sentence_enc = tf.expand_dims(sentence_enc, 0)

    embedding_layer = PositionalEmbeddings(config)
    pos_embed = embedding_layer(sentence_enc)
    print(f'Positional embed shape: {pos_embed.shape}')

if __name__ == '__main__':
    test()

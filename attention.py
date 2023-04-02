'''
Attention
'''


import tensorflow as tf
import tiktoken


def scaled_dot_product(query, key, value, mask=None):
        # calculate similarities between query and keys
        matmul_qk = tf.matmul(query, key, transpose_b=True)
        if mask is not None:
            matmul_qk = tf.where(mask, matmul_qk, float('-inf'))
        query_dim = tf.cast(query.shape[-1], dtype=tf.float32)
        logits = matmul_qk / tf.math.sqrt(query_dim)

        # get weights and contextualize the word vectors (values)
        attention_weights = tf.nn.softmax(logits, axis=-1)
        return tf.matmul(attention_weights, value)


class AttentionHead(tf.keras.layers.Layer):
    def __init__(self, head_dim, name='attention_head'):
        super(AttentionHead, self).__init__(name=name)
        self.query_dense = tf.keras.layers.Dense(head_dim)
        self.key_dense = tf.keras.layers.Dense(head_dim)
        self.value_dense = tf.keras.layers.Dense(head_dim)
        
    def call(self, input):
        q = self.query_dense(input)
        k = self.key_dense(input)
        v = self.value_dense(input)
        attn = scaled_dot_product(q, k, v)
        return attn
    

class MultiAttentionHead(tf.keras.layers.Layer):
    def __init__(self, config, name='multi_attention_head'):
        super(MultiAttentionHead, self).__init__(name=name)
        head_dim = config['hidden_size'] // config['num_heads']
        self.heads = [AttentionHead(head_dim) for _ in range(config['num_heads'])]
        self.output_layer = tf.keras.layers.Dense(config['hidden_size'])

    def call(self, input):
        x = [h(input) for h in self.heads]
        x = tf.concat(x, axis=-1)
        x = self.output_layer(x)
        return x


def test():
    config = {'num_heads': 4, 
              'vocab_size': 50257,
              'hidden_size': 128}
    sentence = 'It is a good day to have lunch'
    encoder = tiktoken.get_encoding('p50k_base')
    sentence_enc = tf.convert_to_tensor(encoder.encode(sentence) + [0, 0], dtype=tf.int64)
    sentence_enc = tf.expand_dims(sentence_enc, 0)
    embeds = tf.keras.layers.Embedding(config['vocab_size'], 
                                       config['hidden_size'])(sentence_enc)
    print(f'Embeds shape: {embeds.shape}\n{embeds}')
    
    head = AttentionHead(config['hidden_size'])
    atten_out = head(embeds)
    print(f'Attention out shape: {atten_out.shape}\n{atten_out}')

    head = MultiAttentionHead(config)
    atten_out = head(embeds)
    print(f'Multi head attention out shape: {atten_out.shape}\n{atten_out}')

if __name__ == '__main__':
    test()

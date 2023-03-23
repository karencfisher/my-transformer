'''
Attention
'''


import tensorflow as tf
import tiktoken


def scaled_dot_product(query, key, value):
        # calculate similarities between query and keys
        matmul_qk = tf.matmul(query, key, transpose_b=True)
        query_dim = tf.cast(query.shape[-1], dtype=tf.float32)
        logits = matmul_qk / tf.math.sqrt(query_dim)

        # get weights and contextualize the word vectors (values)
        attention_weights = tf.nn.softmax(logits, axis=-1)
        return tf.matmul(attention_weights, value)


class AttentionHead(tf.keras.layers.Layer):
    def __init__(self, embed_dim, head_dim, name='attention_head'):
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
    def __init__(self, embed_dim, num_heads, name='multi_attention_head'):
        super(MultiAttentionHead, self).__init__(name=name)
        head_dim = embed_dim // num_heads
        self.heads = [AttentionHead(embed_dim, head_dim) for _ in range(num_heads)]
        self.output_layer = tf.keras.layers.Dense(embed_dim)

    def call(self, input):
        x = tf.concat([h(input) for h in self.heads])
        x = self.output_layer(x)
        return x


def test():
    sentence = 'It is a good day to have lunch'
    encoder = tiktoken.get_encoding('p50k_base')
    sentence_enc = tf.convert_to_tensor(encoder.encode(sentence), dtype=tf.int64)
    sentence_enc = tf.expand_dims(sentence_enc, 0)
    embeds = tf.keras.layers.Embedding(50257, 64)(sentence_enc)
    print(f'Embeds shape: {embeds.shape}\n{embeds}')

    embed_dim = embeds.shape[-1]
    head = AttentionHead(embed_dim, embed_dim)
    atten_out = head(embeds)
    print(f'Attention out shape: {atten_out.shape}\n{atten_out}')

    

if __name__ == '__main__':
    test()

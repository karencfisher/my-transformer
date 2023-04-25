'''
Attention
'''


import tensorflow as tf


def scaled_dot_product(query, key, value, mask=None):
        # calculate similarities between query and keys
        matmul_qk = tf.matmul(query, key, transpose_b=True)
        query_dim = tf.cast(query.shape[-1], dtype=tf.float32)
        logits = matmul_qk / tf.math.sqrt(query_dim)
        # if mask is not None:
        #     logits = tf.where(mask, logits, float('-inf'))

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
        x, m = input
        q = self.query_dense(x)
        k = self.key_dense(x)
        v = self.value_dense(x)
        attn = scaled_dot_product(q, k, v, mask=m)
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


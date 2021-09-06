import tensorflow as tf

class EmbeddingLayer(tf.keras.Model):

  def __init__(self, vocab_size, embedding_dim):
    super(EmbeddingLayer, self).__init__()
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

  def call(self, inputs, **kwargs):
    return self.embedding(inputs) 

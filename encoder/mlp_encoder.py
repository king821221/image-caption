import tensorflow as tf

from encoder.encoder import Encoder


class MlpEncoder(Encoder):

  def __init__(self, embedding_dim, activation=tf.nn.relu, dropout=0.0, name='mlp_encoder'):
    super(MlpEncoder, self).__init__(name=name)

    # shape after fc == (batch_size, feature_length, embedding_dim)
    self.fc = tf.keras.layers.Dense(embedding_dim, name='{}/fc'.format(self.name))
    self.activation = activation
    self.dropout = tf.keras.layers.Dropout(dropout)

  def call(self, inputs, **kwargs):
    x = self.fc(inputs)
    x = self.activation(x)
    training = kwargs['training']
    x = self.dropout(x, training=training)
    return x

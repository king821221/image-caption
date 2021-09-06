import tensorflow as tf


class Encoder(tf.keras.Model):

  # Encoder model
  def __init__(self, name='encoder'):
    super(Encoder, self).__init__(name=name)

  def call(self, inputs, **kwargs):
      pass

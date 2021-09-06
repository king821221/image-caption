import tensorflow as tf

class Decoder(tf.keras.Model):

  def __init__(self, vocab_size, data_type, name='decoder'):
    super(Decoder, self).__init__(name=name)
    self.vocab_size = vocab_size
    self.data_type = data_type

  def call(self, inputs, **kwargs):
    pass

  def reset_state(self, target, feat_d, **kwargs):
    pass

  def copy(self):
    pass

  def create_variables(self):
    pass

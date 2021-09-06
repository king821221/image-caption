import tensorflow as tf


class FeatureProjState(tf.keras.Model):

  def __init__(self, units, name):
    super(FeatureProjState, self).__init__()
    self.fc = tf.keras.layers.Dense(units, name=name)

  def __call__(self, features, mask=None):
    # features: [batch_size, length, embedding dim]
    # mask: [batch_size, length]

    # feature_mean: [batch_size, embedding dim]
    feature_mean = tf.reduce_mean(features, axis=-2)

    if mask is not None:
      # mask_exp: [batch_size, length, 1]
      mask_exp = tf.expand_dims(mask, -1)  
      # feature_mul_mask : [batch_size, length, embedding dim]
      feature_mul_mask = features * mask_exp 
      # feature_mask_sum : [batch_size, embedding dim]
      feature_mask_sum = tf.reduce_sum(feature_mul_mask, axis=-2)
      # mask_sum: [batch_size, 1]
      mask_sum = tf.reduce_sum(mask, axis=-1, keepdims=True)
      # mask_sum: [batch_size, embedding dim]
      mask_sum = tf.tile(mask_sum, [1, feature_mask_sum.shape[-1]]) 
      # feature_mean: [batch_size, embedding dim]
      feature_mean = tf.where(
        tf.equal(mask_sum, 0),
        tf.zeros_like(feature_mask_sum, dtype=feature_mask_sum.dtype),
        feature_mask_sum) 

    feature_as_state = self.fc(feature_mean)

    return feature_as_state 



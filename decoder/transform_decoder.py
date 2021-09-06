import tensorflow as tf

from decoder.decoder import Decoder
from decoder.ffn_layer import FeedForwardNetwork
from attention.attention import get_decoder_self_attention_bias
from attention.attention import MultiHeadAttention
from embedding.position_embedding import RelativePositionEmbedding

class PrePostProcessingWrapper(tf.keras.layers.Layer):
  """Wrapper class that applies layer pre-processing and post-processing."""

  def __init__(self, layer, params):
    super(PrePostProcessingWrapper, self).__init__()
    self.layer = layer
    self.params = params

  def build(self, input_shape):
    # Create normalization layer
    self.layer_norm = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, dtype="float32")
    self.postprocess_dropout = tf.keras.layers.Dropout(
        self.params["layer_postprocess_dropout"])
    super(PrePostProcessingWrapper, self).build(input_shape)

  def get_config(self):
    return {
        "params": self.params,
    }

  def call(self, x, *args, **kwargs):
    """Calls wrapped layer with same parameters."""
    # Preprocessing: apply layer normalization
    training = kwargs["training"]

    y = self.layer_norm(x)

    # Get layer output
    y = self.layer(y, *args, **kwargs)

    # Postprocessing: apply dropout and residual connection
    y = self.postprocess_dropout(y, training=training)
    return x + y


class DecoderStack(tf.keras.layers.Layer):
  """Transformer decoder stack.

  Like the encoder stack, the decoder stack is made up of N identical layers.
  Each layer is composed of the sublayers:
    1. Self-attention layer
    2. Multi-headed attention layer combining encoder outputs with results from
       the previous self-attention layer.
    3. Feedforward network (2 fully-connected layers)
  """

  def __init__(self, params):
    super(DecoderStack, self).__init__()
    self.params = params
    self.layers = []

  def build(self, input_shape):
    """Builds the decoder stack."""
    params = self.params
    for _ in range(params["num_hidden_layers"]):
      self_attention_layer = MultiHeadAttention(
          self.params["hidden_size"],
          self.params["num_heads"],
          dropout=self.params["attention_dropout"])
      enc_dec_attention_layer = MultiHeadAttention(
          self.params["hidden_size"],
          self.params["num_heads"],
          dropout=self.params["attention_dropout"])
      feed_forward_network = FeedForwardNetwork(
          self.params["hidden_size"],
          self.params["filter_size"],
          self.params["relu_dropout"])

      self.layers.append([
          PrePostProcessingWrapper(self_attention_layer, params),
          PrePostProcessingWrapper(enc_dec_attention_layer, params),
          PrePostProcessingWrapper(feed_forward_network, params)
      ])
    self.output_normalization = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, dtype="float32")
    super(DecoderStack, self).build(input_shape)

  def get_config(self):
    return {
        "params": self.params,
    }

  def call(self,
           decoder_inputs,
           encoder_outputs,
           decoder_self_attention_bias,
           attention_bias,
           training,
           decode_loop_step=None):
    """Return the output of the decoder layer stacks.

    Args:
      decoder_inputs: A tensor with shape [batch_size, target_length,
        hidden_size].
      encoder_outputs: A tensor with shape [batch_size, input_length,
        hidden_size]
      decoder_self_attention_bias: A tensor with shape [1, 1, target_len,
        target_length], the bias for decoder self-attention layer.
      attention_bias: A tensor with shape [batch_size, 1, 1, input_length], the
        bias for encoder-decoder attention layer.
      training: A bool, whether in training mode or not.
      cache: (Used for fast decoding) A nested dictionary storing previous
        decoder self-attention values. The items are:
          {layer_n: {"k": A tensor with shape [batch_size, i, key_channels],
                     "v": A tensor with shape [batch_size, i, value_channels]},
                       ...}
      decode_loop_step: An integer, the step number of the decoding loop. Used
        only for autoregressive inference on TPU.

    Returns:
      Output of decoder layer stack.
      float32 tensor with shape [batch_size, target_length, hidden_size]
    """
    print("decoder layer inputs {} enc outputs {} decode_bias {}".format(
      decoder_inputs.shape, encoder_outputs.shape, decoder_self_attention_bias.shape
    ))

    for n, layer in enumerate(self.layers):
      self_attention_layer = layer[0]
      enc_dec_attention_layer = layer[1]
      feed_forward_network = layer[2]

      # Run inputs through the sublayers.
      layer_name = "layer_%d" % n
      with tf.name_scope(layer_name):
        with tf.name_scope("self_attention"):
          decoder_inputs = self_attention_layer(
              decoder_inputs,
              decoder_inputs,
              attention_mask=decoder_self_attention_bias,
              training=training)
        with tf.name_scope("encdec_attention"):
          decoder_inputs = enc_dec_attention_layer(
              decoder_inputs,
              encoder_outputs,
              attention_mask=attention_bias,
              training=training)
        with tf.name_scope("ffn"):
          decoder_inputs = feed_forward_network(
              decoder_inputs, training=training)

    return self.output_normalization(decoder_inputs)


class TransformDecoder(Decoder):

  def __init__(self, params, vocab_size, data_type=tf.int32):
    super(TransformDecoder, self).__init__(vocab_size, data_type)
    hidden_size = params['hidden_size']
    self.embedding = tf.keras.layers.Embedding(vocab_size, hidden_size)
    self.fc2 = tf.keras.layers.Dense(vocab_size)
    self.decoder_stack = DecoderStack(params)
    self.position_embedding = RelativePositionEmbedding(
        hidden_size=hidden_size)
    self.decode_droput = tf.keras.layers.Dropout(
        params["layer_postprocess_dropout"])

  def call(self, inputs, **kwargs):
    dec_input, feat_d, hidden = inputs

    features = feat_d['features']

    training = kwargs['training'] 

    # h_state: [batch_size, length]
    prev_decoded = hidden[0]
    
    targets = tf.concat([prev_decoded, dec_input], -1)

    print("transformer decode targets {} features {}".format(targets.shape, features.shape))

    return self.decode(targets, features, **kwargs) 

  def decode(self,
             targets,
             encoder_outputs,
             attention_bias=None,
             **kwargs):
    """Generate logits for each value in the target sequence.

    Args:
      targets: target values for the output sequence. int tensor with shape
        [batch_size, target_length]
      encoder_outputs: continuous representation of input sequence. float
      tensor
        with shape [batch_size, input_length, hidden_size]
      attention_bias: float tensor with shape [batch_size, 1, 1,
      input_length]
    Returns:
      float32 tensor with shape [batch_size, target_length, vocab_size]
    """
    training = kwargs['training']
    with tf.name_scope("decode"):
        # Prepare inputs to decoder layers by shifting targets, adding
        # positional
        # encoding and applying dropout.
        decoder_inputs = self.embedding(targets)
        with tf.name_scope("add_pos_encoding"):
          length = decoder_inputs.shape[1]
          pos_encoding = self.position_embedding(decoder_inputs)
          decoder_inputs += pos_encoding
        decoder_inputs = self.decode_droput(decoder_inputs,
                                            training=training)

        # Run values
        decoder_self_attention_bias = get_decoder_self_attention_bias(
            length, dtype=decoder_inputs.dtype)
        print("transformer decoder_inputs {} bias {}".format(
          decoder_inputs.shape, decoder_self_attention_bias.shape
        ))
        outputs = self.decoder_stack(
          decoder_inputs,
          encoder_outputs,
          decoder_self_attention_bias,
          attention_bias,
          training=training)
        print("outputs shape {}".format(outputs.shape))
        logits = self.fc2(outputs)
        predictions = logits[:, -1, :]
        state = targets
        print("transformer decoder logits {} state {} ".format(logits.shape, state.shape))
        return predictions, [state], None, outputs[:, -1, :] 

  def reset_state(self, target, features):
    batch_size = target.shape[0]
    return [tf.zeros((batch_size, 1), dtype=self.data_type)]

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import numpy as np
import tensorflow as tf

from beam_search.beam_search import beam_search_step_batch
from beam_search.beam_search import create_initial_beam_state_batch
from beam_search.beam_search import BeamSearchConfig
from beam_search.beam_search import choose_top_k
from beam_search.beam_search import loc_optimal_beam_path_batch

def standard_normalize(values, axes=(0,)):
  """Standard normalizes values `values`.

  Args:
    values: Tensor with values to be standardized.
    axes: Axes used to compute mean and variances.

  Returns:
    Standardized values (values - mean(values[axes])) / std(values[axes]).
  """
  values_mean, values_var = tf.nn.moments(x=values, axes=axes, keepdims=True)
  epsilon = np.finfo(values.dtype.as_numpy_dtype).eps
  normalized_values = ((values - values_mean) / (tf.sqrt(values_var) + epsilon))
  return normalized_values

def sequence_length_fn(sequences, sequence_mask, keepdims=False):
    sequences_mask_flag = tf.cast(sequences > sequence_mask, tf.int32)
    return tf.reduce_sum(sequences_mask_flag, axis=-1, keepdims=keepdims)

class SequenceSampler(object):

    def __init__(self,
                 action_network,
                 start_token,
                 eos_token,
                 pad_token = 0,
                 log_prob_fn = tf.nn.log_softmax,
                 dtype = tf.int32):
      self.action_network = action_network 
      self.start_token = start_token
      self.pad_token = pad_token
      self.eos_token = eos_token
      self.log_prob_fn = log_prob_fn
      self.data_type = dtype

    def __call__(self, inputs, **kwargs):
      features, sequences = inputs

      batch_size = sequences.shape[0]

      start_token_batch = tf.zeros((batch_size, 1), dtype=self.data_type) + \
                          self.start_token

      eos_token_batch = tf.zeros((batch_size, 1), dtype=self.data_type) + \
                        self.eos_token

      pad_token_batch = tf.zeros((batch_size, 1), dtype=self.data_type) + \
                        self.pad_token

      # sequence_length: [batch_size, 1]
      sequence_length = sequence_length_fn(sequences, self.pad_token, keepdims=True)

      samples = [start_token_batch]

      state = self.action_network.reset_state(sequences, {"features": features}, **kwargs)
      # dec_input: [batch_size]
      dec_input = start_token_batch[:, 0]

      for time_step in range(1, sequences.shape[1]):
        # predictions: [batch_size, vocab_size]
        predictions, state = self.action_network((dec_input,
                                                  {"features": features},
                                                  state),
                                                 **kwargs)
        vocab_size = predictions.shape[-1]

        p_mask = tf.one_hot(
          self.pad_token,
          vocab_size,
          dtype=predictions.dtype,
          off_value=0.,
          on_value=tf.float32.min)
        p_mask = tf.expand_dims(p_mask, 0)

        s_mask = tf.one_hot(
          self.start_token,
          vocab_size,
          dtype=predictions.dtype,
          off_value=0.,
          on_value=tf.float32.min)
        s_mask = tf.expand_dims(s_mask, 0)

        e_mask = tf.one_hot(
          self.eos_token,
          vocab_size,
          dtype=predictions.dtype,
          off_value=0.,
          on_value=tf.float32.min)
        e_mask = tf.expand_dims(e_mask, 0)

        predictions += p_mask
        predictions += s_mask
        predictions += e_mask

        log_probs = self.log_prob_fn(predictions)
        logging.debug("sample log probs {} {} {} {}".format(time_step, tf.reduce_min(log_probs), tf.reduce_max(log_probs), tf.reduce_mean(log_probs)))
        # sample: [batch_size, 1]
        sample = tf.random.categorical(log_probs, 1)

        sample = tf.cast(sample, self.data_type)

        sample = tf.where(tf.equal(time_step + 1, sequence_length),
                          eos_token_batch,
                          sample)

        sample = tf.where(tf.greater_equal(time_step, sequence_length),
                          pad_token_batch,
                          sample)

        sample = tf.cast(sample, self.data_type)

        samples.append(sample)

        dec_input = tf.squeeze(sample, -1)

      # samples: [batch_size, sequence_length]
      samples = tf.concat(samples, -1)
      return samples

class BaselineSequeneBeamSearcher(object):

  def __init__(self,
               actor_network,
               beam_width,
               vocab_size,
               max_length,
               start_token,
               eos_token,
               pad_token=0):
    self.actor_network = actor_network
    self.beam_width = beam_width
    self.vocab_size = vocab_size
    self.max_length = max_length
    self.start_token = start_token
    self.eos_token = eos_token
    self.pad_token = pad_token

  def __call__(self, features, sequences, **kwargs):
    batch_size = features.shape[0]

    # [beam_width, 1]
    dec_input = tf.expand_dims([self.start_token] * self.beam_width, 1)

    # [batch_size*beam_width, 1]
    dec_input = tf.tile(dec_input, [batch_size, 1])

    # [batch_size*beam_width, feat_length, emb_dim]
    features = tf.tile(features, [self.beam_width, 1, 1])

    # hidden: [[batch_size * beam_width, units]]
    dummy_tgt = tf.zeros_like(dec_input)
    hidden = self.actor_network.reset_state(dummy_tgt, {'features':features}, **kwargs)

    bs_state = create_initial_beam_state_batch(batch_size, self.beam_width)
    bs_config = BeamSearchConfig(beam_width=self.beam_width,
                                 vocab_size=self.vocab_size,
                                 eos_token=self.eos_token,
                                 length_penalty_weight=1.0,
                                 choose_successors_fn=choose_top_k)


    bs_sequence = []

    max_length = self.max_length
    if max_length <= 0:
      max_length = sequences.shape[1]

    #sequence_length: [batch_size * beam_width, 1]
    sequence_length = sequence_length_fn(sequences, self.pad_token, keepdims=True)
    sequence_length = tf.tile(sequence_length, [self.beam_width, 1])

    vocab_size = self.vocab_size

    s_mask = tf.one_hot(
      self.start_token,
      vocab_size,
      off_value=0.,
      on_value=tf.float32.min)
    s_mask = tf.expand_dims(s_mask, 0)

    e_mask = tf.one_hot(
      self.eos_token,
      vocab_size,
      on_value=0.,
      off_value=tf.float32.min)
    e_mask = tf.expand_dims(e_mask, 0)

    p_mask = tf.one_hot(
      self.pad_token,
      vocab_size,
      on_value=0.,
      off_value=tf.float32.min)
    p_mask = tf.expand_dims(p_mask, 0)

    for i in range(1, max_length):
      # predictions: [1 * beam_width, vocab_size]
      # hidden: [1 * beam_width, state dim]
      # attention_weight: [1 * beam_width, feature length]

      # predictions: [batch_size * beam_width, vocab_size]
      # hidden: [[batch_size * beam_width, units]]
      # attention_weights: [batch_size * beam_width, T, T]
      # context_vec: [batch_size * beam_width, units]
      predictions, hidden = self.actor_network((dec_input,
                                               {"features": features},
                                               hidden),
                                               training=False)
      # mask out <s>
      predictions += tf.cast(s_mask, predictions.dtype)

      #eos_flag : [batch_size * beam_width, 1]
      if self.max_length <= 0:
        eos_flag = tf.equal(i+1, sequence_length)
        eos_flag = tf.cast(eos_flag, predictions.dtype)
        eos_mask = e_mask * eos_flag
        predictions += tf.cast(eos_mask, predictions.dtype)

      #pad_flag : [batch_size * beam_width, 1]
      if self.max_length <= 0:
        pad_flag = tf.greater(i+1, sequence_length)
        pad_flag = tf.cast(pad_flag, predictions.dtype)
        pad_mask = p_mask * pad_flag
        predictions += tf.cast(pad_mask, predictions.dtype)

      predictions = tf.reshape(predictions,
                               [-1, self.beam_width, predictions.shape[-1]])

      bs_out, bs_state = beam_search_step_batch(i,
                                                predictions,
                                                bs_state,
                                                bs_config)

      bs_sequence.append((bs_out, bs_state))

      # Grow from beam out's predicted id
      # dec_input: [batch_size * beam_width, 1]
      dec_input = tf.reshape(bs_out.predicted_ids, [-1, 1])

    decode_sequence_vec = list(loc_optimal_beam_path_batch(bs_sequence))

    decode_sequence = tf.concat(decode_sequence_vec, -1)

    s_batch = tf.zeros_like(decode_sequence, dtype=decode_sequence.dtype)[:, 0:1]
    s_batch += self.start_token

    decode_sequence = tf.concat([s_batch, decode_sequence], axis=-1)

    return decode_sequence

class SequenceBeamSearchSampler(object):

    def __init__(self,
                 actor_network,
                 beam_width,
                 vocab_size,
                 start_token,
                 eos_token,
                 pad_token = 0,
                 dtype = tf.int32):
      self.eos_token = eos_token
      self.searcher = BaselineSequeneBeamSearcher(actor_network,
                                                  beam_width,
                                                  vocab_size,
                                                  0,
                                                  start_token,
                                                  eos_token)
    def __call__(self, inputs, **kwargs):
      features, sequences = inputs
      baseline_sequence = self.searcher(features, sequences, **kwargs)
      return baseline_sequence

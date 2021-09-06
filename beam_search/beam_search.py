# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""In-Graph Beam Search Implementation.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
import logging
import tensorflow as tf
from tensorflow.python.util import nest  # pylint: disable=E0611

class BeamSearchState(
    namedtuple("BeamSearchState", ["log_probs", "finished", "lengths"])):
  """State for a single step of beam search.

  Args:
    log_probs: The current log probabilities of all beams
    finished: A boolean vector that specifies which beams are finished
    lengths: Lengths of all beams
  """
  pass


class BeamSearchStepOutput(
    namedtuple("BeamSearchStepOutput",
               ["scores", "predicted_ids", "beam_parent_ids"])):
  """Outputs for a single step of beam search.

  Args:
    scores: Score for each beam, a float32 vector of shape [beam_width]
    predicted_ids: predictions for this step step, an int32 vector of shape [beam_width]
    beam_parent_ids: an int32 vector containing the beam indices of the
      continued beams from the previous step, a vector of shape [beam_width]
  """
  pass


class BeamSearchConfig(
    namedtuple("BeamSearchConfig", [
        "beam_width", "vocab_size", "eos_token", "length_penalty_weight",
        "choose_successors_fn"
    ])):
  """Configuration object for beam search.

  Args:
    beam_width: Number of beams to use, an integer
    vocab_size: Output vocabulary size
    eos_token: The id of the EOS token, used to mark beams as "done"
    length_penalty_weight: Weight for the length penalty factor. 0.0 disables
      the penalty.
    choose_successors_fn: A function used to choose beam successors based
      on their scores. Maps from (scores, config) => (chosen scores, chosen_ids)
  """
  pass

def create_initial_beam_state(beam_width):
  """Creates an instance of `BeamState` that can be used on the first
  call to `beam_step`.

  Args:
    config: A BeamSearchConfig

  Returns:
    An instance of `BeamState`.
  """
  return BeamSearchState(
      log_probs=tf.zeros([beam_width]),
      finished=tf.zeros([beam_width], dtype=tf.bool),
      lengths=tf.zeros([beam_width], dtype=tf.int32))

def create_initial_beam_state_batch(batch_size, beam_width):
  """Creates an instance of `BeamState` that can be used on the first
  call to `beam_step`.

  Args:
    config: A BeamSearchConfig

  Returns:
    An instance of `BeamState`.
  """
  return BeamSearchState(
      log_probs=tf.zeros([batch_size, beam_width]),
      finished=tf.zeros([batch_size, beam_width], dtype=tf.bool),
      lengths=tf.zeros([batch_size, beam_width], dtype=tf.int32))

def combine_beam_state_batch(beam_state_left, beam_state_right, axis=-1):
    log_probs = tf.concat([beam_state_left.log_probs,
                           beam_state_right.log_probs],
                          axis=axis)
    finished = tf.concat([beam_state_left.finished,
                          beam_state_right.finished],
                          axis=axis)
    lengths = tf.concat([beam_state_left.lengths,
                         beam_state_right.lengths],
                         axis=axis)
    return BeamSearchState(
      log_probs=log_probs,
      finished=finished,
      lengths=lengths)

def length_penalty(sequence_lengths, penalty_factor):
  """Calculates the length penalty according to
  https://arxiv.org/abs/1609.08144

   Args:
    sequence_lengths: The sequence length of all hypotheses, a tensor
      of shape [beam_size, vocab_size].
    penalty_factor: A scalar that weights the length penalty.

  Returns:
    The length penalty factor, a tensor fo shape [beam_size].
   """
  return tf.math.divide((5. + tf.compat.v1.to_float(sequence_lengths))**penalty_factor, (5. + 1.)
                **penalty_factor)


def hyp_score(log_probs, sequence_lengths, length_penalty_weight):
  """Calculates scores for beam search hypotheses.
  """

  # Calculate the length penality
  length_penality_ = length_penalty(
      sequence_lengths=sequence_lengths,
      penalty_factor=length_penalty_weight)

  score = log_probs / length_penality_
  return score


def choose_top_k(scores_flat, top_k):
  """Chooses the top-k beams as successors.
  """
  next_beam_scores, word_indices = tf.nn.top_k(scores_flat, k=top_k)
  return next_beam_scores, word_indices


def nest_map(inputs, map_fn, name=None):
  """Applies a function to (possibly nested) tuple of tensors.
  """
  if nest.is_sequence(inputs):
    inputs_flat = nest.flatten(inputs)
    y_flat = [map_fn(_) for _ in inputs_flat]
    outputs = nest.pack_sequence_as(inputs, y_flat)
  else:
    outputs = map_fn(inputs)
  if name:
    outputs = tf.identity(outputs, name=name)
  return outputs


def mask_probs(probs, eos_token, finished):
  """Masks log probabilities such that finished beams
  allocate all probability mass to eos. Unfinished beams remain unchanged.

  Args:
    probs: Log probabiltiies of shape `[beam_width, vocab_size]`
    eos_token: An int32 id corresponding to the EOS token to allocate
      probability to
    finished: A boolean tensor of shape `[beam_width]` that specifies which
      elements in the beam are finished already.

  Returns:
    A tensor of shape `[beam_width, vocab_size]`, where unfinished beams
    stay unchanged and finished beams are replaced with a tensor that has all
    probability on the EOS token.
  """
  vocab_size = tf.shape(probs)[-1]
  finished_mask = tf.expand_dims(
      (1. - tf.cast(finished, tf.float32)), -1)
  # These examples are not finished and we leave them
  # if not finished, finished_mask = 1; if finished, finished_mask = 0
  non_finished_examples = finished_mask * probs
  # All finished examples are replaced with a vector that has all
  # probability on EOS, for entry mapping to eos_token, the one-hot value is 0
  # for entry mapping to non_eos_token, the one-hot value is float.min
  finished_row = tf.one_hot(
      eos_token,
      vocab_size,
      dtype=tf.float32,
      on_value=0.,
      off_value=tf.float32.min)
  finished_examples = (1. - finished_mask) * finished_row
  return finished_examples + non_finished_examples


def beam_search_step(time_, logits, beam_state, config):
  """Performs a single step of Beam Search Decoding.

  Args:
    time_: Beam search time step, should start at 0. At time 0 we assume
      that all beams are equal and consider only the first beam for
      continuations.
    logits: Logits at the current time step. A tensor of shape `[beam_width, vocab_size]`
    beam_state: Current state of the beam search. An instance of `BeamState`
    config: An instance of `BeamSearchConfig`

  Returns:
    A new beam state.
  """

  # Calculate the current lengths of the predictions
  prediction_lengths = beam_state.lengths
  previously_finished = beam_state.finished

  eos_token_id = config.eos_token
  beam_width = config.beam_width
  vocab_size = config.vocab_size
  length_penalty_weight = config.length_penalty_weight

  # Calculate the total log probs for the new hypotheses
  # probs: [beam_width, vocab_size]
  probs = tf.nn.log_softmax(logits)
  logging.debug("probs {}".format(probs))
  probs = mask_probs(probs, eos_token_id, previously_finished)
  logging.debug("mask probs {}".format(probs))
  total_probs = tf.expand_dims(beam_state.log_probs, -1) + probs
  logging.debug("total probs {}".format(total_probs))

  # Calculate the continuation lengths
  # We add 1 to all continuations that are not EOS and were not
  # finished previously
  lengths_to_add = tf.one_hot([eos_token_id] * beam_width,
                              vocab_size, 0, 1)
  add_mask = (1 - tf.cast(previously_finished, tf.int32))
  lengths_to_add = tf.expand_dims(add_mask, 1) * lengths_to_add
  new_prediction_lengths = tf.expand_dims(prediction_lengths,
                                          1) + lengths_to_add
  logging.debug("new prediction length {}".format(new_prediction_lengths))

  # Calculate the scores for each beam
  scores = hyp_score(
      log_probs=total_probs,
      sequence_lengths=new_prediction_lengths,
      length_penalty_weight=length_penalty_weight
  )

  logging.debug("hyp_score {}".format(scores))
  # scores_flat has shape [beam_width * vocab_size]
  scores_flat = tf.reshape(scores, [-1])
  # During the first time step we only consider the initial beam
  scores_flat = tf.cond(
      tf.constant(time_) > 0, lambda: scores_flat, lambda: scores[0])

  # Pick the next beams according to the specified successors function
  # this means top_k from the <beam_width * vocab_size>
  choose_successors_fn = config.choose_successors_fn
  next_beam_scores, word_indices = choose_successors_fn(scores_flat,
                                                        beam_width)
  next_beam_scores.set_shape([beam_width])
  word_indices.set_shape([beam_width])
  logging.debug("next_beam_scores {} word indices {}".format(next_beam_scores, word_indices))
  # Pick out the probs, beam_ids, and states according to the chosen predictions
  total_probs_flat = tf.reshape(total_probs, [-1], name="total_probs_flat")
  next_beam_probs = tf.gather(total_probs_flat, word_indices)
  next_beam_probs.set_shape([beam_width])
  next_word_ids = tf.compat.v1.mod(word_indices, vocab_size)
  next_beam_ids = tf.compat.v1.div(word_indices, vocab_size)
  logging.debug("next_beam_probs {} next_word_ids {} next_beam_ids {}".format(next_beam_probs, next_word_ids, next_beam_ids))

  # Append new ids to current predictions
  next_finished = tf.logical_or(
      tf.gather(beam_state.finished, next_beam_ids),
      tf.equal(next_word_ids, eos_token_id))
  logging.debug("next_finished {}".format(next_finished))
  # Calculate the length of the next predictions.
  # 1. Finished beams remain unchanged
  # 2. Beams that are now finished (EOS predicted) remain unchanged
  # 3. Beams that are not yet finished have their length increased by 1
  lengths_to_add = tf.compat.v1.to_int32(tf.not_equal(next_word_ids, eos_token_id))
  lengths_to_add = (1 - tf.compat.v1.to_int32(next_finished)) * lengths_to_add
  next_prediction_len = tf.gather(beam_state.lengths, next_beam_ids)
  next_prediction_len += lengths_to_add

  next_state = BeamSearchState(
      log_probs=next_beam_probs, # keep the accumulated log probs
      lengths=next_prediction_len,
      finished=next_finished)

  output = BeamSearchStepOutput(
      scores=next_beam_scores,
      predicted_ids=next_word_ids,
      beam_parent_ids=next_beam_ids)

  return output, next_state

def loc_optimal_beam_path(beam_sequence):
    decode_sequence = []

    if len(beam_sequence) == 0:
        return decode_sequence

    bs_out, bs_state = beam_sequence[-1]
    predicted_ids = bs_out.predicted_ids
    predicted_beam_ids = bs_out.beam_parent_ids
    predicted_scores = bs_out.scores
    finished = bs_state.finished
    finished = tf.cast(finished, predicted_scores.dtype)
    finished_at_least = tf.reduce_max(finished)
    finished = (1.0 - finished) * -1e9
    logging.debug("predict ids {}".format(predicted_ids)) 
    logging.debug("predict beam ids {}".format(predicted_beam_ids)) 
    logging.debug("predict beam scores {}".format(predicted_scores)) 
    logging.debug("predict beam finished {}".format(finished)) 
    predicted_scores = tf.cond(finished_at_least > 0,
                               lambda: predicted_scores + finished,
                               lambda: predicted_scores)
    logging.debug("predicted scores last {}".format(predicted_scores))
    max_pred_score_idx = tf.argmax(predicted_scores)
    logging.debug("max pred idx {}".format(max_pred_score_idx))
    max_predict_id = tf.gather(predicted_ids, max_pred_score_idx)
    max_predict_beam_id = tf.gather(predicted_beam_ids, max_pred_score_idx)
    logging.debug("max pred id {}".format(max_predict_id))
    logging.debug("max pred beam id {}".format(max_predict_beam_id))

    decode_sequence.append(max_predict_id)

    for idx in range(len(beam_sequence) - 2, -1, -1):
        bs_out, bs_state = beam_sequence[idx]
        predicted_ids = bs_out.predicted_ids
        predicted_beam_ids = bs_out.beam_parent_ids
        logging.debug("idx {} predict ids {} predict beam ids {}".format(idx, predicted_ids, predicted_beam_ids))
        logging.debug("idx {} max_predict_beam {}".format(idx, max_predict_beam_id))
        max_predict_id = tf.gather(predicted_ids, max_predict_beam_id)
        max_predict_beam_id = tf.gather(predicted_beam_ids, max_predict_beam_id)
        logging.debug("idx {} max_predict_id {}".format(idx, max_predict_id))
        logging.debug("idx {} next_max_predict_beam {}".format(idx, max_predict_beam_id))
        decode_sequence.append(max_predict_id)

    decode_sequence = reversed(decode_sequence)

    return decode_sequence

def batch_gather(a, b):
  a_dim = tf.shape(a)[-1]
  b_dim = tf.shape(b)[-1]
  def apply_gather(x):
    xp, yp = tf.split(x, [a_dim, b_dim], axis=-1)
    return tf.gather(xp, tf.cast(yp, tf.int32))

  a = tf.cast(a, dtype=tf.float32)
  b = tf.cast(b, dtype=tf.float32)
  stacked = tf.concat([a, b], axis=-1)
  gathered = tf.map_fn(apply_gather, stacked)
  return tf.stack(gathered, axis=0)

def beam_search_step_batch(time_, logits, beam_state, config, prob_fn = lambda x: tf.nn.log_softmax(x), start_offset=0):
  """Performs a single step of Beam Search Decoding.

  Args:
    time_: Beam search time step, should start at 0. At time 0 we assume
      that all beams are equal and consider only the first beam for
      continuations.
    logits: Logits at the current time step. A tensor of shape `[batch_size, beam_width, vocab_size]`
    beam_state: Current state of the beam search. An instance of `BeamState`
    config: An instance of `BeamSearchConfig`

  Returns:
    A new beam state.
  """

  #print("beam_search batch input beam_state {} logits {}".format(
    #beam_state, logits))

  batch_size = tf.shape(logits)[0]
  beam_width = config.beam_width 
  vocab_size = config.vocab_size

  logging.info("beam width  {} vocab size {}".format(beam_width, vocab_size))

  # Calculate the current lengths of the predictions
  prediction_lengths = beam_state.lengths # [batch_size, beam_width]
  previously_finished = beam_state.finished # [batch_siz, *beam_width]

  # logits: [batch_size * beam_width, vocab_size]
  logits_r = tf.reshape(logits, [-1, vocab_size])

  # previously_finished: [batch_size * beam_width]
  previously_finished_r = tf.reshape(previously_finished, [-1])

  # Calculate the total log probs for the new hypotheses
  # probs: [batch_size * beam_width, vocab_size]
  probs = prob_fn(logits_r)
  #probs = tf.compat.v1.Print(probs, [tf.shape(probs), probs], message = 'probs', summarize = 100)
  logging.debug("probs {}".format(probs))

  probs = mask_probs(probs, config.eos_token, previously_finished_r)
  logging.debug("mask_probs {}".format(probs.shape))
  #probs = tf.compat.v1.Print(probs, [tf.shape(probs), probs], message = 'mask_probs', summarize = 100)
  # probs: [batch_size * beam_width, vocab_size]
  # beam_state.log_probs:  [batch_size * beam_width]
  #total_probs: [batch_size * beam_width, vocab_size]
  total_probs = tf.reshape(beam_state.log_probs, [-1, 1]) + probs
  logging.debug("total_probs {}".format(total_probs.shape))
  #total_probs: [batch_size, beam_width, vocab_size]
  total_probs = tf.reshape(total_probs, [batch_size, -1, vocab_size])
  logging.debug("total_probs {}".format(total_probs.shape))
  #total_probs = tf.compat.v1.Print(total_probs, [tf.shape(total_probs), total_probs], message = 'total_probs', summarize=100)

  # Calculate the continuation lengths
  # We add 1 to all continuations that are not EOS and were not
  # finished previously
  lengths_to_add = tf.one_hot([config.eos_token] * logits.shape[1],
                              config.vocab_size, 0, 1)
  # lengths_to_add: [1,beam_width, vocab_size]
  lengths_to_add = tf.expand_dims(lengths_to_add, 0) # [1, beam_width, vocab_size]
  logging.debug("lengths_to_add {}".format(lengths_to_add.shape))
  #lengths_to_add = tf.compat.v1.Print(lengths_to_add, [tf.shape(lengths_to_add), lengths_to_add], message = 'lengths_to_add', summarize=100)

  # add_mask: [batch_size, beam_width, 1]
  logging.debug("finished : {}".format(previously_finished.shape))
  add_mask = (1 - tf.cast(previously_finished, tf.int32)) # [batch_size*beam_width]
  add_mask = tf.expand_dims(add_mask, -1)
  logging.debug("add_mask {}".format(add_mask.shape))
  #add_mask = tf.compat.v1.Print(add_mask, [tf.shape(add_mask), add_mask], message = 'add_mask', summarize=100)
  # add_mask: [batch_size, beam_width, 1]
  # lengths_to_add: [batch_size, beam_width, vocab_size]
  lengths_to_add = add_mask * lengths_to_add
  
  #lengths_to_add = tf.compat.v1.Print(lengths_to_add, [tf.shape(lengths_to_add), lengths_to_add], message = 'lengths_to_add_mask', summarize=100)
  logging.debug("lengths_to_add {}".format(lengths_to_add.shape))

  # new_predicted_lengths: [batch_size, beam_width, vocab_size]
  new_prediction_lengths = tf.expand_dims(prediction_lengths, -1) + lengths_to_add
  logging.debug("new_prediction_lengths {}".format(new_prediction_lengths.shape))
  #new_prediction_lengths = tf.compat.v1.Print(new_prediction_lengths, [tf.shape(new_prediction_lengths), new_prediction_lengths], message = 'new_prediction_lengths', summarize=100)

  # Calculate the scores for each beam
  scores = hyp_score(
      log_probs=total_probs,
      sequence_lengths=new_prediction_lengths,
      length_penalty_weight=config.length_penalty_weight)
  # scores: [batch_size, beam_width, vocab_size]
  logging.debug("hyp_score {}".format(scores.shape))
  #scores = tf.compat.v1.Print(scores, [tf.shape(scores), scores], message = 'scores', summarize=100)

  # scores_flat has shape [batch_size, beam_width * vocab_size]
  scores_flat = tf.reshape(scores, [batch_size, -1])
  # During the first time step we only consider the initial beam with shape
  # [batch_size, vocab_size]
  scores_flat = tf.cond(
      tf.convert_to_tensor(time_) > 0, lambda: scores_flat, lambda: scores[:, start_offset, :])
  logging.debug("scores_flat {}".format(scores_flat.shape))
  #scores_flat = tf.compat.v1.Print(scores_flat, [tf.shape(scores_flat), scores_flat], message = 'scores_flat', summarize=100)

  # Pick the next beams according to the specified successors function
  # this means top_k from the <batch_size, beam_width * vocab_size>
  next_beam_scores, word_indices = config.choose_successors_fn(scores_flat,
                                                               beam_width)
  logging.debug("next_beam_scores {} word_indices {}".format(
      next_beam_scores.shape, word_indices.shape
  ))
  #next_beam_scores = tf.compat.v1.Print(next_beam_scores, [tf.shape(next_beam_scores), next_beam_scores], summarize=100, message = 'next_beam_scores')
  #word_indices = tf.compat.v1.Print(word_indices, [tf.shape(word_indices), word_indices], summarize=100, message = 'word_indices')

  # next_beam_scores: [batch_size, beam_width]
  # word_indices: [batch_size, beam_width]

  # Pick out the probs, beam_ids, and states according to the chosen predictions
  # total_probs_flat: [batch_size, beam_width * vocab_size]
  # total_probs is the un-normalized scores of each new beam
  total_probs_flat = tf.reshape(total_probs, [batch_size, -1], name="total_probs_flat")

  next_beam_probs = batch_gather(total_probs_flat, word_indices)
  #next_beam_probs = tf.reshape(next_beam_probs, [batch_size, beam_width])
  # next_beam_probs: [batch_size, beam_width]

  # next_word_ids: [batch_size, beam_width]
  # next_beam_ids: [batch_size, beam_width]
  next_word_ids = tf.compat.v1.mod(word_indices, vocab_size)
  next_beam_ids = tf.compat.v1.div(word_indices, vocab_size)

  logging.debug("total_probs_flat {} next_beam_probs {} next_word_ids {} next_beam_ids {}".format(
     total_probs_flat.shape, next_beam_probs.shape, next_word_ids.shape, next_beam_ids.shape
  ))
  #next_beam_probs = tf.compat.v1.Print(next_beam_probs,
   #                        [tf.shape(next_beam_probs), next_beam_probs],
   #                        summarize=100,
   #                        message='next_beam_probs')
  #next_word_ids = tf.compat.v1.Print(next_word_ids,
   #                        [tf.shape(next_word_ids), next_word_ids],
   #                        summarize=100,
   #                        message='next_word_ids')
  #next_beam_ids = tf.compat.v1.Print(next_beam_ids,
   #                        [tf.shape(next_beam_ids), next_beam_ids],
   #                        summarize=100, message = 'next_beam_ids')

  # Append new ids to current predictions
  # beam_state.finished: [batch_size, beam_width]
  # beam_state_finished: [batch_size, beam_width]
  beam_state_finished = batch_gather(beam_state.finished, next_beam_ids)
  beam_state_finished = tf.cast(beam_state_finished, tf.bool)
  #beam_state_finished = tf.reshape(beam_state_finished, [batch_size, beam_width])
  logging.debug("beam_state_finished {}".format(beam_state_finished.shape))
  next_finished = tf.logical_or(
      beam_state_finished,
      tf.equal(next_word_ids, config.eos_token))
  # next_finished: [batch_size, beam_width]

  # Calculate the length of the next predictions.
  # 1. Finished beams remain unchanged
  # 2. Beams that are now finished (EOS predicted) remain unchanged
  # 3. Beams that are not yet finished have their length increased by 1

  lengths_to_add = tf.cast(tf.not_equal(next_word_ids, config.eos_token), tf.int32)
  lengths_to_add = (1 - tf.cast(next_finished, tf.int32)) * lengths_to_add
  logging.debug("lengths_to_add {}".format(lengths_to_add.shape))
  #lengths_to_add = tf.compat.v1.Print(lengths_to_add, [tf.shape(lengths_to_add), lengths_to_add], summarize=100, message = 'lengths_to_add')
  # lengths_to_add: [batch_size, beam_width]

  next_prediction_len = batch_gather(beam_state.lengths, next_beam_ids)
  next_prediction_len = tf.cast(next_prediction_len, tf.int32)
  #next_prediction_len = tf.reshape(next_prediction_len, [batch_size, beam_width])
  # next_prediction_len: [batch_size, beam_width]
  logging.debug("next_prediction_len {}".format(next_prediction_len.shape))
  next_prediction_len += lengths_to_add
  # next_prediction_len: [batch_size, beam_width]
  #next_prediction_len = tf.compat.v1.Print(next_prediction_len, [tf.shape(next_prediction_len), next_prediction_len], summarize=100, message = 'next_prediction_len')

  next_state = BeamSearchState(
      log_probs=next_beam_probs,
      lengths=next_prediction_len,
      finished=next_finished)

  output = BeamSearchStepOutput(
      scores=next_beam_scores,
      predicted_ids=next_word_ids,
      beam_parent_ids=next_beam_ids)

  return output, next_state

def loc_optimal_beam_path_batch(beam_sequence):
    decode_sequence = []

    if len(beam_sequence) == 0:
        return decode_sequence

    bs_out, bs_state = beam_sequence[-1]
    # predicted_ids: [batch_size, beam_width]
    predicted_ids = bs_out.predicted_ids
    # predicted_beam_ids: [batch_size, beam_width]
    predicted_beam_ids = bs_out.beam_parent_ids
    # predicted_scores: [batch_size, beam_width]
    predicted_scores = bs_out.scores
    # finished: [batch_size, beam_width]
    finished = bs_state.finished
    finished = tf.cast(finished, predicted_scores.dtype)

    beam_width = predicted_ids.shape[-1]

    # finished_at_least: [batch_size, 1]
    finished_at_least = tf.reduce_max(finished, axis=-1, keepdims=True)
    # finished_at_least: [batch_size, beam_width]
    finished_at_least = tf.tile(finished_at_least, [1, beam_width])
    # finished_mask: [batch_size, beam_width]
    finished_mask = (1.0 - finished) * -1e9
    logging.debug("predict ids {}".format(predicted_ids))
    logging.debug("predict beam ids {}".format(predicted_beam_ids))
    logging.debug("predict beam scores {}".format(predicted_scores))
    logging.debug("predict beam finished {}".format(finished))
    logging.debug("predict beam finished_mask {}".format(finished_mask))

    # predicted_scores: [batch_size, beam_width]
    predicted_scores = tf.where(finished_at_least > 0,
                                predicted_scores + finished_mask,
                                predicted_scores)
    logging.debug("predicted scores last {}".format(predicted_scores))
    # max_pred_score_idx: [batch_size,]
    max_pred_score_idx = tf.argmax(predicted_scores, axis=-1)
    # max_pred_score_idx: [batch_size, 1]
    max_pred_score_idx = tf.expand_dims(max_pred_score_idx, -1)
    logging.debug("max pred idx {}".format(max_pred_score_idx))
    # max_predict_id: [batch_size, 1]
    max_predict_id = batch_gather(predicted_ids, max_pred_score_idx)
    # max_predict_beam_id: [batch_size, 1]
    max_predict_beam_id = batch_gather(predicted_beam_ids, max_pred_score_idx)
    logging.debug("max pred id {}".format(max_predict_id))
    logging.debug("max pred beam id {}".format(max_predict_beam_id))

    decode_sequence.append(max_predict_id)

    for idx in range(len(beam_sequence) - 2, -1, -1):
        bs_out, bs_state = beam_sequence[idx]
        predicted_ids = bs_out.predicted_ids
        predicted_beam_ids = bs_out.beam_parent_ids
        logging.debug("idx {} predict ids {} predict beam ids {}"
                      .format(idx, predicted_ids, predicted_beam_ids))
        logging.debug("idx {} max_predict_beam {}"
                      .format(idx, max_predict_beam_id))
        max_predict_id = batch_gather(predicted_ids, max_predict_beam_id)
        max_predict_beam_id = batch_gather(predicted_beam_ids,
                                           max_predict_beam_id)
        logging.debug("idx {} max_predict_id {}".format(idx, max_predict_id))
        logging.debug("idx {} next_max_predict_beam {}"
                      .format(idx, max_predict_beam_id))
        decode_sequence.append(max_predict_id)

    decode_sequence = reversed(decode_sequence)

    return decode_sequence

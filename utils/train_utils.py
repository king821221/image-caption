from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import numpy as np
import nltk
import tensorflow as tf

from attention.attention import _NEG_INF_FP16
from attention.attention import _NEG_INF_FP32

from beam_search.beam_search import beam_search_step_batch
from beam_search.beam_search import create_initial_beam_state_batch
from beam_search.beam_search import BeamSearchConfig
from beam_search.beam_search import choose_top_k
from beam_search.beam_search import combine_beam_state_batch
from beam_search.beam_search import batch_gather
from utils.tf_utils import soft_variables_update
from utils.tf_utils import Periodically
from utils.tf_utils import clip_gradient_norms
from utils.tf_utils import add_variables_summaries
from utils.tf_utils import tf_summary
from utils.reinforce import ReinforceValueMlpNetwork
from utils.reinforce import SequenceBleuRewardFn
from utils.reinforce import SequencePerStepRewardFn 
from utils.reinforce import RnnDecoderNetwork 
from utils.reinforce import ReinforceAgent
from utils.agent_utils import BaselineSequeneBeamSearcher
from utils.reinforce import SelfCriticalReinforceAgent 
from utils.actor_critic import ActorCriticAgent 
from utils.agent_utils import SequenceSampler 
from utils.agent_utils import SequenceBeamSearchSampler 

eos_token = 4
start_token = 3

def bleu_score_py(real, predict):
  bleu_vec = []

  print("real {} predict {}".format(np.shape(real), np.shape(predict)))

  for idx in range(len(real)):
      rl = list(real[idx])
      pr = list(predict[idx])

      new_rl = []

      for e in rl:
        if e == eos_token:
          break
        if e != start_token:
          new_rl.append(e)

      new_pr = []

      for e in pr:
        if e == eos_token:
          break
        if e != start_token:
          new_pr.append(e)

      bleu = nltk.translate.bleu_score.sentence_bleu([new_rl], new_pr)
      bleu_vec.append(bleu)

  return np.array(bleu_vec).astype(np.float32)

@tf.function
def bleu_score_tf(real, pred):
  y = tf.numpy_function(bleu_score_py, [real, pred], tf.float32)
  return y

class GoldProbFn(object):

  def __init__(self, gold_prob_mask):
    self.gold_prob_mask = gold_prob_mask

  def __call__(self, logits, **kwargs):
    probs = tf.nn.log_softmax(logits)
    return probs + self.gold_prob_mask

def train_step_bso(src,
                   target,
                   encoder,
                   decoder,
                   step,
                   optimizer,
                   beam_width,
                   vocab_size,
                   start_token_id=start_token,
                   eos_token_id=eos_token,
                   length_penalty_weight=1.2,
                   hinge_thr=0.0):
  # Follow paper: Sequence-to-Sequence Learning as Beam-Search Optimization

  gold_beam_width = 1

  dec_input = tf.expand_dims([start_token_id] * target.shape[0], axis=1)
  # dec_input_gold: [batch_size*gold_beam_width, 1]
  dec_input_gold = tf.tile(dec_input, [gold_beam_width, 1])
  # dec_input_sch: [batch_size*beam_width, 1]
  dec_input_sch = tf.tile(dec_input, [beam_width, 1])
  # dec_input_gold_sch: [batch_size*(beam_width+gold_beam_width), 1]
  dec_input_gold_sch = tf.concat([dec_input_gold, dec_input_sch], axis=0)

  print("beam_width {} dec input gold {}  dec input sch {} dec input gold sch {}".format(beam_width, dec_input_gold.shape, dec_input_sch.shape, dec_input_gold_sch.shape))

  gold_beam_config = BeamSearchConfig(beam_width=gold_beam_width,
                                      vocab_size=vocab_size,
                                      eos_token=eos_token_id,
                                      length_penalty_weight=length_penalty_weight,
                                      choose_successors_fn=choose_top_k)

  beam_config = BeamSearchConfig(beam_width=beam_width,
                                 vocab_size=vocab_size,
                                 eos_token=eos_token_id,
                                 length_penalty_weight=length_penalty_weight,
                                 choose_successors_fn=choose_top_k)

  print("train_bso beam_config {}".format(beam_config))

  sequence_loss = 0
  token_loss = 0
  token_loss_mask = 0

  with tf.GradientTape() as tape:
    features = encoder(src, training=True) # [B, L, H]

    # hidden: [[batch_size, units]]
    hidden = decoder.reset_state(target, {"features": features})
    # hidden_gold: [batch_size * gold_beam_width, units]
    hidden_gold = [tf.tile(h_state, [gold_beam_width, 1]) for h_state in hidden]

    for h in hidden_gold:
      print("gold hidden state {}".format(h.shape))

    # hidden_sch: [batch_size * beam_width, units]
    hidden_sch = [tf.tile(h_state, [beam_width, 1]) for h_state in hidden]

    for h in hidden_sch:
      print("sch hidden state {}".format(h.shape))

    # hidden_gold_sch: [batch_size * (beam_width + gold_beam_width), units]
    hidden_gold_sch = [tf.concat([h_gold, h_sch], 0) for h_gold, h_sch in zip(hidden_gold, hidden_sch)]

    for h in hidden_gold_sch:
      print("gold_sch hidden state {}".format(h.shape))

    # features_gold: [batch_size * gold_beam_width, L, H]
    features_gold = tf.tile(features, [gold_beam_width, 1, 1])
    # features_sch: [batch_size * beam_width, L, H]
    features_sch = tf.tile(features, [beam_width, 1, 1])

    # features_gold_sch: [batch_size *(beam_width+gold_beam_width), L, H]
    features_gold_sch = tf.concat([features_gold, features_sch], axis=0)

    tf_summary('train_step_features', features, step=step, mode='histogram')

    print("features gold {} features sch {} gold_sch {}".format(
      features_gold.shape, features_sch.shape, features_gold_sch.shape
    ))

    # gold_beam_mask: [batch_size, gold_beam_width]
    gold_beam_mask = tf.zeros((target.shape[0], gold_beam_width),
                              dtype=tf.int32)
    # gold_beam_mask: [batch_size*gold_beam_width, 1]
    gold_beam_mask = tf.reshape(gold_beam_mask, [-1, 1])

    # sch_beam_mask: [batch_size, beam_width]
    sch_beam_mask = tf.ones((target.shape[0], beam_width),
                            dtype=tf.int32)
    # sch_beam_mask: [batch_size*beam_width, 1]
    sch_beam_mask = tf.reshape(sch_beam_mask, [-1, 1])

    # gold_sch_beam_mask: [batch_size * (beam_width + gold_beam_width), 1]
    gold_sch_beam_mask = tf.concat([gold_beam_mask, sch_beam_mask], axis=0)

    print("gold_beam_mask {} sch_beam_mask {} gold_sch_beam_mask {}".format(
       gold_beam_mask.shape, sch_beam_mask.shape, gold_sch_beam_mask.shape
    ))

    gold_beam_state = create_initial_beam_state_batch(target.shape[0],
                                                      gold_beam_width)

    logging.debug("gold_beam_state {}".format(gold_beam_state))

    gold_sch_beam_state = create_initial_beam_state_batch(target.shape[0],
                                                          gold_beam_width+beam_width)

    logging.debug("gold_sch_beam_state {}".format(gold_sch_beam_state))

    beam_sequence = []

    for i in range(1, target.shape[1]):
      # predictions: [batch_size * (beam_width+gold_beam_width), vocab_size]
      # hidden_beam_gold: [batch_size * (beam_width+gold_beam_width), units]
      predictions, hidden_gold_sch, attention_weights, ctx_vec = \
        decoder((dec_input_gold_sch,
                 {"features": features_gold_sch},
                 hidden_gold_sch),
                training=True)

      logging.debug("step {} predictions {} hidden {}".format(i, predictions, hidden_gold_sch))

      preds = tf.reshape(predictions, [-1, gold_beam_width + beam_width, predictions.shape[-1]]) 

      # tgt_gold_preds: [batch_size, gold_beam_width, vocab_size]
      tgt_gold_preds = preds[:, 0:gold_beam_width, :]

      logging.debug("tgt_gold_preds {}".format(tgt_gold_preds.shape))

      #tgt_gold_preds= tf.compat.v1.Print(tgt_gold_preds, [tf.shape(tgt_gold_preds), tgt_gold_preds], summarize=100, message = 'tgt_gold_preds')
      tf_summary('tgt_gold_preds', tgt_gold_preds, step=step, mode='histogram')

      # tgt: [batch_size]
      tgt = target[:, i]

      # tgt_gold: [batch_size * gold_beam_width]
      tgt_gold = tf.tile(tgt, [gold_beam_width])

      tgt_gold = tf.reshape(tgt_gold, [-1, gold_beam_width])

      # tgt_onehot: [batch_size , gold_beam_width, vocab_size]
      tgt_onehot = tf.one_hot(
        tgt_gold,
        vocab_size,
        dtype=tgt_gold_preds.dtype,
        on_value=0.,
        off_value=tf.float32.min)

      logging.debug("tgt_onehot {}".format(tgt_onehot.shape))

      # tgt_predictions: [batch_size * gold_beam_width, vocab_size]
      # tgt_onehot = tf.compat.v1.Print(tgt_onehot, [tf.shape(tgt_onehot), tgt_onehot], summarize=100, message = 'tgt_onehot')

      gold_prob_fn = GoldProbFn(tgt_onehot)

      gold_beam_out, gold_beam_state = beam_search_step_batch(i,
                                                              tgt_gold_preds,
                                                              gold_beam_state,
                                                              gold_beam_config,
                                                              prob_fn=gold_prob_fn)

      neg_inf = _NEG_INF_FP16 if predictions.dtype == tf.float16 else _NEG_INF_FP32

      gold_sch_beam_mask = tf.cast(gold_sch_beam_mask, predictions.dtype)

      logging.debug("step {} gold_sch_beam_mask {}".format(i, gold_sch_beam_mask))

      # gold_sch_beam_mask = tf.compat.v1.Print(gold_sch_beam_mask, [tf.shape(gold_sch_beam_mask), gold_sch_beam_mask], summarize=100, message = 'gold_sch_beam_mask')

      predictions+=(1-gold_sch_beam_mask) * neg_inf

      tf_summary('decode_step_prediction', predictions, step=step, mode='histogram')

      # is_last_step: [batch_size, 1]
      is_last_step = tf.equal(target[:, i], eos_token_id)
      is_last_step = tf.expand_dims(is_last_step, -1)

      # is_last_step = tf.compat.v1.Print(is_last_step, [tf.shape(is_last_step), is_last_step], summarize=100, message = 'is_last_step')
      # mask: [batch_size, 1]
      mask = tf.cast(tf.greater(target[:, i], 0), tf.int32)
      mask = tf.expand_dims(mask, -1)
      # mask = tf.compat.v1.Print(mask, [tf.shape(mask), mask], summarize=100, message = 'mask')

      predictions = tf.reshape(predictions,
         [-1,
          gold_beam_width+beam_width,
          predictions.shape[-1]])

      #predictions = tf.compat.v1.Print(predictions, [tf.shape(predictions), predictions], summarize=100, message = 'predictions')

      # each element in sch_beam_out and sch_beam_state
      # has beam_width
      sch_beam_out, sch_beam_state = beam_search_step_batch(i,
                                                            predictions,
                                                            gold_sch_beam_state,
                                                            beam_config,
                                                            start_offset=1)

      logging.debug("step {} sch_beam_out {} sch_beam_state {}".format(step, sch_beam_out, sch_beam_state))

      # sch_beam_scores: [batch_size, beam_width]
      sch_beam_scores = sch_beam_out.scores

      tf_summary('decode_step_sch_beam_scores', sch_beam_scores, step=step, mode='histogram')

      # gold_beam_scores: [batch_size, gold_beam_width==1]
      gold_beam_scores = gold_beam_out.scores

      tf_summary('decode_step_gold_beam_scores', gold_beam_scores, step=step, mode='histogram')

      # beam_scores_min: [batch_size, 1]
      beam_scores_min = tf.reduce_min(sch_beam_scores, axis=-1, keepdims=True)

      # beam_scores_max: [batch_size, 1]
      beam_scores_max = tf.reduce_max(sch_beam_scores, axis=-1, keepdims=True)

      # beam_scores_baseline: [batch_size, 1]
      beam_scores_baseline = tf.where(is_last_step,
                                      beam_scores_max,
                                      beam_scores_min)

      tf_summary('decode_step_beam_scores_baseline', beam_scores_baseline, step=step, mode='histogram')
      # violations: [batch_size, gold_beam_width==1]
      # gold_beam_scores = tf.compat.v1.Print(gold_beam_scores, [tf.shape(gold_beam_scores), tf.reduce_mean(gold_beam_scores), gold_beam_scores], summarize=100, message = 'gold_beam_scores')
      # sch_beam_scores = tf.compat.v1.Print(sch_beam_scores, [tf.shape(sch_beam_scores), tf.reduce_mean(sch_beam_scores), sch_beam_scores], summarize=100, message = 'sch_beam_scores')
      # beam_scores_baseline = tf.compat.v1.Print(beam_scores_baseline, [tf.shape(beam_scores_baseline), tf.reduce_mean(beam_scores_baseline), beam_scores_baseline], summarize=100, message = 'beam_scores_baseline')
      violations = tf.less(gold_beam_scores, beam_scores_baseline+hinge_thr)
      tf_summary('decode_step_violations', violations, step=step, mode='histogram')

      #violations = tf.compat.v1.Print(violations, [tf.shape(violations), violations], summarize=100, message = 'violations')
      # If there is violation,
      # At next time step, expand from the gold beam
      # otherwise, expand from the search beam

      sch_predicted_ids = sch_beam_out.predicted_ids
      gold_predicted_ids = gold_beam_out.predicted_ids

      #sch_predicted_ids = tf.compat.v1.Print(sch_predicted_ids, [tf.shape(sch_predicted_ids), tf.reduce_min(sch_predicted_ids), tf.reduce_max(sch_predicted_ids), sch_predicted_ids], summarize=100, message = 'sch_predicted_ids')
      #gold_predicted_ids = tf.compat.v1.Print(gold_predicted_ids, [tf.shape(gold_predicted_ids), tf.reduce_min(gold_predicted_ids), tf.reduce_max(gold_predicted_ids), gold_predicted_ids], summarize=100, message = 'gold_predicted_ids')
      dec_input_gold_sch = tf.reshape(
          tf.concat([gold_predicted_ids, sch_predicted_ids], -1), (-1, 1)
      )

      gold_sch_beam_state = combine_beam_state_batch(
          gold_beam_state, sch_beam_state
      )

      violations = tf.cast(violations, tf.int32)

      # gold_beam_mask: [batch_size, gold_beam_width==1]
      gold_beam_mask = tf.ones((target.shape[0], gold_beam_width),
                                dtype=tf.int32) * violations

      # gold_beam_mask: [batch_size*gold_beam_width, 1]
      gold_beam_mask = tf.reshape(gold_beam_mask, [-1, 1])

      # sch_beam_mask: [batch_size, beam_width]
      sch_beam_mask = tf.ones((target.shape[0], beam_width),
                              dtype=tf.int32) * (1 - violations)
      # sch_beam_mask: [batch_size*beam_width, 1]
      sch_beam_mask = tf.reshape(sch_beam_mask, [-1, 1])

      # gold_sch_beam_mask: [batch_size * (beam_width + gold_beam_width), 1]
      gold_sch_beam_mask = tf.concat([gold_beam_mask, sch_beam_mask], axis=0)

      beam_sequence_e_info = (
          sch_beam_out,
          sch_beam_state,
          gold_beam_out,
          gold_beam_state,
          violations
      )

      # loss: [batch_size, gold_beam_width]
      loss = tf.nn.relu(beam_scores_baseline + hinge_thr - gold_beam_scores)
      #loss = tf.compat.v1.Print(loss, [tf.shape(loss), loss], summarize=100, message = 'loss')
      mask = tf.cast(mask, loss.dtype)
      loss *= mask

      tf_summary('decode_step_loss', loss, step=step, mode='histogram')

      #print("step {} loss {}".format(step, loss))

      sequence_loss += tf.reduce_mean(loss)
      token_loss += tf.reduce_sum(loss)
      token_loss_mask += tf.reduce_sum(mask)

      beam_sequence.append(beam_sequence_e_info)

  trainable_variables = encoder.trainable_variables + \
                        decoder.trainable_variables

  logging.debug("train encoder vars {}".format(encoder.trainable_variables))
  logging.debug("train decoder vars {}".format(decoder.trainable_variables))

  gradients = tape.gradient(sequence_loss, trainable_variables)

  optimizer.apply_gradients(zip(gradients, trainable_variables))

  return sequence_loss, token_loss, trainable_variables, gradients

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred, mask_val=0):
  mask = tf.math.logical_not(tf.math.equal(real, mask_val))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)

  loss_ *= mask

  return loss_, mask

def train_step_reinforce(src,
                         target,
                         encoder,
                         decoder,
                         optimizer,
                         start_token,
                         eos_token,
                         **kwargs):
  # Follow paper:Sequence Level Training with Recurrent Neural Networks
  pad_token = kwargs.get('pad_token') or 0

  value_network = ReinforceValueMlpNetwork(name='reinforce_value_network')
  action_network = RnnDecoderNetwork(decoder)
  sequence_reward_fn = SequencePerStepRewardFn(SequenceBleuRewardFn())
  sequence_sampler = SequenceSampler(action_network,
                                     start_token,
                                     eos_token,
                                     dtype=target.dtype,
                                     **kwargs) 

  agent = ReinforceAgent(action_network,
                         sequence_reward_fn,
                         encoder,
                         optimizer,
                         sequence_sampler,
                         start_token,
                         eos_token,
                         pad_token = pad_token,
                         value_network = value_network,
                         gradient_clipping = 5.0,
                         entropy_regularization = 0.8,
                         name = 'reinforce_sequence_predict') 

  return agent((src, target), **kwargs)
 
def train_step_self_critic(src,
                           target,
                           encoder,
                           decoder,
                           optimizer,
                           start_token,
                           eos_token,
                           vocab_size,
                           **kwargs):
  # Follow paper:Sequence Level Training with Recurrent Neural Networks
  pad_token = kwargs.get('pad_token') or 0
  beam_width = kwargs.get('beam_width') or 5
  max_predict_length = kwargs.get('max_predict_length') or 0

  action_network = RnnDecoderNetwork(decoder)
  sequence_reward_fn = SequenceBleuRewardFn()
  sequence_sampler = SequenceSampler(action_network,
                                     start_token,
                                     eos_token,
                                     pad_token=pad_token,
                                     dtype=target.dtype) 

  baseline_sequence_gen = BaselineSequeneBeamSearcher(action_network,
                                                      beam_width,
                                                      vocab_size,
                                                      max_predict_length,
                                                      start_token,
                                                      eos_token)

  agent = SelfCriticalReinforceAgent(action_network,
                                     sequence_reward_fn,
                                     encoder,
                                     optimizer,
                                     sequence_sampler,
                                     baseline_sequence_gen,
                                     start_token,
                                     eos_token,
                                     pad_token = pad_token,
                                     gradient_clipping = 5.0,
                                     entropy_regularization = 0.8,
                                     name = 'self_critic_sequence_predict') 

  return agent((src, target), **kwargs)
  
def train_step_actor_critic(src,
                            target,
                            embedding_layer,
                            critic_network,
                            target_critic_network,
                            actor_network,
                            delayed_actor_network,
                            src_sequence_encoder,
                            target_sequence_encoder,
                            sequence_reward_fn,
                            actor_optimizer,
                            critic_optimizer,
                            start_token,
                            eos_token,
                            vocab_size,
                            step,
                            **kwargs):
  # Follow paper:AN ACTOR-CRITIC ALGORITHM FOR SEQUENCE PREDICTION
  critic_loss_weight = 1.0
  if 'critic_loss_weight' in kwargs:
    critic_loss_weight = kwargs.get('critic_loss_weight')
  actor_loss_weight = 1.0
  if 'actor_loss_weight' in kwargs:
    actor_loss_weight = kwargs.get('actor_loss_weight')
  ct_penalize_coef = kwargs.get('ct_penalize_coef') or 0.01
  cl_penalize_coef = kwargs.get('cl_penalize_coef') or 0.01
  td_err_loss_fn = kwargs.get('td_err_loss_fn') or (lambda pred, real : tf.math.pow(pred - real, 2))
  gradient_clipping = kwargs.get('gradient_clipping')
  summarize_grads_and_vars = kwargs.get('summarize_grads_and_vars')
  pad_token = kwargs.get('pad_token') or 0
  beam_width = kwargs.get('beam_width') or 1
  target_update_tau = kwargs.get('target_update_tau') or 0.2
  target_update_period = kwargs.get('target_update_period') or 1

  sequence_sampler_s = SequenceSampler(delayed_actor_network,
                                       start_token,
                                       eos_token,
                                       pad_token=pad_token,
                                       dtype=target.dtype) 
  sequence_sampler_b = SequenceBeamSearchSampler(delayed_actor_network,
                                                 beam_width,
                                                 vocab_size,
                                                 start_token,
                                                 eos_token,
                                                 pad_token=pad_token)

  #sequence_sampler = sequence_sampler_s
  sequence_sampler = sequence_sampler_b

  agent = ActorCriticAgent(embedding_layer,
                           critic_network,
                           target_critic_network,
                           actor_network,
                           delayed_actor_network,
                           actor_optimizer,
                           critic_optimizer,
                           sequence_sampler,
                           src_sequence_encoder,
                           target_sequence_encoder,
                           sequence_reward_fn,
                           td_err_loss_fn,
                           start_token = start_token,
                           eos_token = eos_token,
                           pad_token = pad_token,
                           critic_loss_weight = critic_loss_weight,
                           actor_loss_weight = actor_loss_weight,
                           ct_penalize_coef = ct_penalize_coef,
                           cl_penalize_coef = cl_penalize_coef,
                           target_update_tau= target_update_tau,
                           target_update_period = target_update_period,
                           gradient_clipping = gradient_clipping,
                           summarize_grads_and_vars = summarize_grads_and_vars) 

  loss, actor_loss, critic_loss, trainable_variables, gradients = agent.train(src,
                                                                              target,
                                                                              summary_step=step,
                                                                              **kwargs)

  for vs, vd in zip(actor_network.trainable_variables, delayed_actor_network.trainable_variables):
    print("actor_network var {} {} vs delayed var {} {}".format(vs.name, vs, vd.name, vd))

  return loss, actor_loss, trainable_variables, gradients 


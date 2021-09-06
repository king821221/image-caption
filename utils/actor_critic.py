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
from utils.agent_utils import sequence_length_fn
from utils.agent_utils import SequenceSampler
 

class TargetSequenceEncoder(tf.keras.Model):

  def __init__(self, embedding, cell_units, output_dim, sequence_mask, data_type = tf.float32):
    super(TargetSequenceEncoder, self).__init__()
    self.embedding = embedding 
    self.cell = tf.keras.layers.LSTMCell(cell_units) 
    self.cell_units = cell_units
    self.fc = tf.keras.layers.Dense(output_dim, name='target_sequence_encoder_out_fc')
    self.sequence_mask = sequence_mask
    self.data_type = data_type 

  def call(self, sequences, **kwargs):
    scope = 'target_sequence_encoder_cell'
    sequence_length = sequence_length_fn(sequences, self.sequence_mask)
    initial_state = self.reset_state(sequences)
    sequence_emb = self.embedding(sequences)
    outputs, _ = tf.compat.v1.nn.dynamic_rnn(
      self.cell,
      sequence_emb,
      sequence_length=sequence_length,
      initial_state=initial_state,
      dtype=self.data_type,
      scope=scope
    )
    outputs = self.fc(outputs)
    return outputs

  def reset_state(self, sequences):
    return [tf.zeros((sequences.shape[0], self.cell_units), dtype=self.dtype)] * 2

  def create_variables(self):
    sequence = tf.ones((1, 1))
    self.call(sequence)

class ActorCriticAgent(object):

  def __init__(self,
               embedding_layer,
               critic_network,
               target_critic_network,
               actor_network,
               delayed_actor_network,
               actor_optimizer,
               critic_optimizer,
               sequence_sampler,
               src_feature_encoder,
               target_sequence_encoder,
               sequence_reward_fn,
               td_err_loss_fn,
               start_token,
               eos_token,
               pad_token = 0,
               voc_prob_fn = tf.nn.softmax,
               train_step_counter=None,
               critic_loss_weight=1.0,
               actor_loss_weight=1.0,
               ct_penalize_coef=0.1,
               cl_penalize_coef=0.1,
               target_update_tau=0.1,
               target_update_period=1,
               gradient_clipping=None,
               summarize_grads_and_vars=False):
      self.embedding_layer = embedding_layer 
      self.critic_network = critic_network
      self.target_critic_network = target_critic_network
      self.actor_network = actor_network
      self.delayed_actor_network = delayed_actor_network
      self.actor_optimizer = actor_optimizer
      self.critic_optimizer = critic_optimizer
      self.sequence_sampler = sequence_sampler
      self.src_feature_encoder= src_feature_encoder
      self.target_sequence_encoder = target_sequence_encoder
      self.sequence_reward_fn = sequence_reward_fn 
      self.td_err_loss_fn = td_err_loss_fn
      self.voc_prob_fn = voc_prob_fn
      self.start_token = start_token
      self.eos_token = eos_token
      self.pad_token = pad_token
      self.train_step_counter = train_step_counter
      if train_step_counter is None:
        self.train_step_counter =\
          tf.compat.v1.train.get_or_create_global_step()
      self.critic_loss_weight = critic_loss_weight
      self.actor_loss_weight = actor_loss_weight
      self.ct_penalize_coef = ct_penalize_coef
      self.cl_penalize_coef = cl_penalize_coef
      self.gradient_clipping = gradient_clipping
      self.update_target = self.get_target_updater(
          tau=target_update_tau, period=target_update_period)
      self.summarize_grads_and_vars = summarize_grads_and_vars

  def compute_step_critic_target(self,
                                 sample_ahead_probs,
                                 sample_ahead_critic_val,
                                 rt,
                                 **kwargs):
    # qt: [batch_size]
    # qt = rt + SUM(p'(a|Y'[1..t], X) * Q'(a; Y'[1..t], Y))
    qt = rt + tf.reduce_sum(sample_ahead_probs * sample_ahead_critic_val,
                            axis=-1)
    return qt

  def compute_step_critic_loss(self,
                               critic_val,
                               target_critic_val,
                               sample_critic_predictions,
                               **kwargs):
    # Q(y'[t], Y'[1..t-1], Y) - qt
    # td_err: [batch_size]
    td_err = self.td_err_loss_fn(target_critic_val, critic_val)
    print("td_err {}".format(tf.reduce_mean(td_err)))
    # ct: [batch_size]
    ct = self.reduce_critic_val_variance(sample_critic_predictions, **kwargs)
    print("ct {}".format(tf.reduce_mean(ct)))
    # [batch_size]
    return -1 * (td_err + self.ct_penalize_coef * ct)

  def compute_step_actor_loss(self,
                              sample_prob_dist,
                              critic_action_scores,
                              target_action_prob,
                              **kwargs):
    # sample_dist_v: [batch_size, vocab_size]
    # p(a|Y'[1..t-1], X) * Q(a; Y'[1..t-1], Y)
    sample_dist_v = sample_prob_dist * tf.stop_gradient(critic_action_scores)
    # sample_dist_w: [batch_size]
    sample_dist_w = tf.reduce_sum(sample_dist_v, -1)
    # SUM{p(a|Y'[1..t-1], X) * Q(a; Y'[1..t-1], Y)} + cl * p(y[t]|Y[1..t-1], X)
    # batch_size
    return -1 * (sample_dist_w + target_action_prob * self.cl_penalize_coef)

  def reduce_critic_val_variance(self, critic_vals, **kwargs):
    # C[t] = SUM{(Q(a; Y'[1..t-1]) - mean(Q(b; Y'[1..t-1])))}
    # critic_vals: [batch_size, vocab_size]
    # critic_vals_mean: [batch_size, 1]
    critic_vals_mean = tf.reduce_mean(critic_vals, axis=-1, keepdims=True)
    # critic_vals_deduct: [batch_size, vocab_size]
    critic_vals_deduct = critic_vals - critic_vals_mean
    critic_vals_deduct *= critic_vals_deduct
    # [batch_size]
    return tf.reduce_mean(critic_vals_deduct, axis=-1)

  def get_target_updater(self, tau=1.0, period=1):
    """Performs a soft update of the target network parameters.

    For each weight w_s in the original network, and its corresponding
    weight w_t in the target network, a soft update is:
    w_t = (1- tau) x w_t + tau x ws

    Args:
      tau: A float scalar in [0, 1]. Default `tau=1.0` means hard update.
      period: Step interval at which the target network is updated.

    Returns:
      A callable that performs a soft update of the target network parameters.
    """
    with tf.name_scope('update_target'):

      def update():
        """Update target network."""
        actor_update = soft_variables_update(
            self.actor_network.variables,
            self.delayed_actor_network.variables,
            tau,
            tau_non_trainable=1.0)
        critic_update = soft_variables_update(
            self.critic_network.variables,
            self.target_critic_network.variables,
            tau,
            tau_non_trainable=1.0)
        return tf.group(actor_update, critic_update)

      return Periodically(update, period, 'update_targets')

  def train(self, raw_features, sequences, **kwargs):
    """Returns a train op to update the actor and critic networks.

    This method trains with the provided batched features plus sequence labels.

    Args:
      raw_features: raw features as X in paper: [batch_size, input_length, feature_dim]
      sequences: ground-truth sequence labels as Y in paper: [batch_size, output_length]

    Returns:
      A train_op.

    Raises:
      ValueError: If optimizers are None and no default value was provided to
        the constructor.
    """

    summary_step = kwargs['summary_step']

    batch_size = sequences.shape[0]

    sequence_length = sequences.shape[1]

    trainable_critic_variables = []
    trainable_critic_variables.extend(self.critic_network.trainable_variables)
    trainable_critic_variables.extend(self.target_sequence_encoder.trainable_variables)

    print("critic network {} target sequence encoder {}".format(
      self.critic_network, self.target_sequence_encoder    
    ))
    for var in self.critic_network.trainable_variables:
      print("critic network var {}".format(var.name))
    for var in self.target_sequence_encoder.trainable_variables:
      print("target feature encoder var {}".format(var.name))

    trainable_actor_variables = []
    trainable_actor_variables.extend(self.actor_network.trainable_variables)
    trainable_actor_variables.extend(self.src_feature_encoder.trainable_variables)
    trainable_actor_variables.extend(self.embedding_layer.trainable_variables)

    print("actor network {} src feature encoder {}".format(
      self.actor_network, self.src_feature_encoder
    ))
    for var in self.actor_network.trainable_variables:
      print("actor network var {}".format(var.name))
    for var in self.src_feature_encoder.trainable_variables:
      print("source feature encoder var {}".format(var.name))
    for var in self.embedding_layer.trainable_variables:
      print("embedding layer var {}".format(var.name))


    with tf.GradientTape(watch_accessed_variables=True) as actor_tape, \
         tf.GradientTape(watch_accessed_variables=True) as critic_tape:
      assert trainable_critic_variables, ('No trainable critic variables to '
                                          'optimize.')
      assert trainable_actor_variables, ('No trainable actor variables to '
                                          'optimize.')
      actor_tape.watch(trainable_actor_variables)
      critic_tape.watch(trainable_critic_variables)

      critic_loss = 0.0
      actor_loss = 0.0

      # features: [batch_size, feature_length, embedding_dim]
      features = self.src_feature_encoder(raw_features, **kwargs)

      with tf.name_scope('features'):
        tf.compat.v2.summary.histogram(
         name='raw_features', data=raw_features, step=self.train_step_counter)
        tf.compat.v2.summary.histogram(
          name='features', data=features, step=self.train_step_counter)

      # Sample sequence Y' from X using delayed actor
      # with the same shape as sequences Y
      # sequence_samples: [batch_size, output_length]
      sequence_samples = self.sequence_sampler((features, sequences), **kwargs)
      sequence_samples = tf.stop_gradient(sequence_samples)

      # Input token id at each timestep fed to actor/critic network

      # For Y', actor.
      dec_input_sample = tf.constant([self.start_token] * batch_size)

      # For Y, actor
      dec_input_target = tf.constant([self.start_token] * batch_size)

      # For Y', critic
      dec_input_critic_sample = tf.constant([self.start_token] * batch_size)

      # Hidden state recording previous sequence

      # For Y', actor
      sample_sequence_state = self.actor_network.reset_state(sequence_samples,
                                                             {"features": features},
                                                             **kwargs)

      # For Y, actor
      target_sequence_state = self.actor_network.reset_state(sequences,
                                                             {"features": features},
                                                             **kwargs)

      # For Y', critic
      sample_sequence_critic_state =\
        self.critic_network.reset_state(sequence_samples, {"features": features}, **kwargs)

      # For critic target computation p'(|Y'[1..t])
      dec_input_sample_ahead = tf.constant([self.start_token] * batch_size)
      sample_sequence_ahead_state = self.delayed_actor_network.reset_state(
          sequence_samples,
          {"features": features},
          **kwargs)

      _, sample_sequence_ahead_state = \
        self.delayed_actor_network((dec_input_sample_ahead,
                                    {"features": features},
                                    sample_sequence_ahead_state),
                                   **kwargs)

      # For critic target computation Q'(, Y'[1..t])
      sample_sequence_target_critic_state =\
          self.target_critic_network.reset_state(sequence_samples,
                                                 {"features": features},
                                                 **kwargs)

      # target_sequence_enc_states: [batch_size, target sequence length, hidden units]
      target_sequence_enc_states = self.target_sequence_encoder(sequences, **kwargs)

      _, sample_sequence_target_critic_state =\
        self.target_critic_network((dec_input_sample_ahead,
                                    {"features": target_sequence_enc_states},
                                    sample_sequence_target_critic_state),
                                   **kwargs)

      # actual_sequence_sample_length: [batch_size]
      actual_sequence_sample_length = sequence_length_fn(sequence_samples,
                                                         self.pad_token)

      # rt_vec: [{batch_size}]
      rt_vec = self.sequence_reward_fn(sequences, sequence_samples, **kwargs)

      for time_step in range(1, sequence_length):
         # Y'[1..t-1]
         # dec_input_sample is token y'[t-1]
         # sample_sequence_state is the hidden state of Y' before t-1 in actor network
         # sample_predictions: [batch_size, vocab_size]
         # sample_sequence_state_next: [batch_size, hidden units]
         sample_predictions, sample_sequence_state_next =\
             self.actor_network((dec_input_sample,
                                 {"features": features},
                                 sample_sequence_state),
                                 **kwargs)
         # sample_prob_dist: [batch_size, vocab_size]
         # p(a|Y'[1:t-1], X)
         sample_prob_dist = self.voc_prob_fn(sample_predictions)

         # Y[1..t-1]
         # target_predictions: [batch_size, vocab_size]

         # dec_input_target is token y[t-1]
         # target_sequence_state is the hidden state of Y before t-1 in actor network,
         # target_predictions: [batch_size, vocab_size]
         # target_sequence_state_next: [batch_size, hidden units]
         target_predictions, target_sequence_state_next =\
             self.actor_network((dec_input_target,
                                 {"features": features},
                                 target_sequence_state),
                                 **kwargs)
         # target_prob_dist: [batch_size, vocab_size]
         # p(a|Y[1:t-1], X)
         target_prob_dist = self.voc_prob_fn(target_predictions)

         # Y'[1..t-1] critic
         # dec_input_critic_sample is token y'[t-1]
         # sample_sequence_critic_state is the hidden state of Y' before t-1 in critic network
         # sample_critic_predictions: [batch_size, vocab_size]
         # sample_sequence_critic_state_next: [batch_size, hidden units]
         sample_critic_predictions, sample_sequence_critic_state_next =\
             self.critic_network((dec_input_critic_sample,
                                  {"features": target_sequence_enc_states},
                                  sample_sequence_critic_state),
                                  **kwargs)

         # Move the dec_input_sample to the next position with token y'[t]
         # set sequence hidden state to the next state
         dec_input_sample = sequence_samples[:, time_step]
         sample_sequence_state = sample_sequence_state_next

         # Move the dec_input_target to the next position with token y[t]
         # set sequence hidden state to the next state
         dec_input_target = sequences[:, time_step]
         target_sequence_state = target_sequence_state_next

         # Move the dec_input_critic_sample to the next position with token y'[t]
         # set sequence hidden state to the next state
         dec_input_critic_sample = sequence_samples[:, time_step]
         sample_sequence_critic_state = sample_sequence_critic_state_next

         # target_sample_prob: [batch_size]
         # p(y[t]|Y[1:t-1], X) from p(a|Y[1:t-1], X)
         target_sample_exp = tf.expand_dims(dec_input_target, -1)
         target_sample_prob = batch_gather(target_prob_dist, target_sample_exp)
         target_sample_prob = tf.squeeze(target_sample_prob, -1)

         # critic_sample_pred: [batch_size]
         # Q(y'[t], y'[1:t-1], Y)
         critic_sample_exp = tf.expand_dims(dec_input_critic_sample, -1)
         critic_sample_pred = batch_gather(sample_critic_predictions,
                                           critic_sample_exp)
         critic_sample_pred = tf.squeeze(critic_sample_pred, -1)

         # dec_input_sample_ahead is the token y'[t]
         dec_input_sample_ahead = sequence_samples[:, time_step]
         # sample_ahead_predictions: [batch_size, vocab_size]
         # sample_sequence_ahead_state: [batch_size, hidden units]
         sample_ahead_predictions, sample_sequence_ahead_state = \
             self.delayed_actor_network((dec_input_sample_ahead,
                                         {"features": features},
                                         sample_sequence_ahead_state),
                                         **kwargs)
         # p'(a|Y'[1:t], X)
         sample_ahead_probs = self.voc_prob_fn(sample_critic_predictions)

         # sample_ahead_critic_val: [batch_size, vocab_size]
         # sample_sequence_target_critic_state: [batch_size, hidden units]
         # Q'[a, Y'[1:t], Y]
         sample_ahead_critic_val, sample_sequence_target_critic_state =\
             self.target_critic_network((dec_input_sample_ahead,
                                         {"features": target_sequence_enc_states},
                                         sample_sequence_target_critic_state),
                                        **kwargs)

         # rt: [batch_size]
         rt = rt_vec[time_step - 1]

         # qt: [batch_size]
         qt = self.compute_step_critic_target(sample_ahead_probs,
                                              sample_ahead_critic_val,
                                              rt,
                                              **kwargs)
         print("time_step {} critic_pred {} critic_qt {} rt {} all_critic_predictions {}".format(
           time_step, tf.reduce_mean(critic_sample_pred), tf.reduce_mean(qt), tf.reduce_mean(rt), tf.reduce_mean(sample_critic_predictions)))

         # critic_loss_step: [batch_size]
         critic_loss_step = self.compute_step_critic_loss(
           critic_sample_pred,
           qt,
           sample_critic_predictions,
           **kwargs)
         print("time_step {} critic_loss_step before mask {}".format(
           time_step, tf.reduce_mean(critic_loss_step)
         ))
         sequence_mask_batch = tf.cast(
             tf.less(time_step, actual_sequence_sample_length),
             critic_loss_step.dtype)
         critic_loss_step *= sequence_mask_batch
         print("time_step {} critic_loss_step after mask {}".format(
           time_step, tf.reduce_mean(critic_loss_step)
         ))
         critic_loss += tf.reduce_mean(critic_loss_step)
         with tf.name_scope('critic'):
           tf.compat.v2.summary.histogram(
             name='reward', data=rt, step=self.train_step_counter)
           tf.compat.v2.summary.histogram(
             name='critic_target', data=qt, step=self.train_step_counter)
           tf.compat.v2.summary.histogram(
             name='critic_loss_step', data=critic_loss_step, step=self.train_step_counter)


         # actor_loss_step: [batch_size]
         actor_loss_step = self.compute_step_actor_loss(
           sample_prob_dist,
           sample_critic_predictions,
           target_sample_prob,
           **kwargs)
         print("time_step {} actor_loss_step {}".format(
           time_step, tf.reduce_mean(actor_loss_step)
         ))
         sequence_mask_batch = tf.cast(sequence_mask_batch,
                                       actor_loss_step.dtype)
         actor_loss_step *= sequence_mask_batch
         actor_loss += tf.reduce_mean(actor_loss_step)
         with tf.name_scope('actor'):
           tf.compat.v2.summary.histogram(
             name='actor_loss_step', data=actor_loss_step, step=self.train_step_counter)

      critic_loss *= self.critic_loss_weight

      actor_loss *= self.actor_loss_weight

      print("actor_loss {}".format(actor_loss))
      print("critic_loss {}".format(critic_loss))

      tf.debugging.check_numerics(critic_loss, 'Critic loss is inf or nan.')
      critic_grads = critic_tape.gradient(critic_loss, trainable_critic_variables)
      self.apply_gradients(critic_grads,
                           trainable_critic_variables,
                           self.critic_optimizer)

      tf.debugging.check_numerics(actor_loss, 'Actor loss is inf or nan.')
      actor_grads = actor_tape.gradient(actor_loss, trainable_actor_variables)
      self.apply_gradients(actor_grads,
                           trainable_actor_variables,
                           self.actor_optimizer)

      with tf.name_scope('Losses'):
        tf.compat.v2.summary.scalar(
         name='critic_loss', data=critic_loss, step=self.train_step_counter)
        tf.compat.v2.summary.scalar(
          name='actor_loss', data=actor_loss, step=self.train_step_counter)

    self.train_step_counter.assign_add(1)
    self.update_target()

    total_loss = critic_loss + actor_loss

    output_trainable_vars = []
    output_trainable_vars.extend(trainable_critic_variables)
    output_trainable_vars.extend(trainable_actor_variables)

    output_grads = []
    output_grads.extend(critic_grads)
    output_grads.extend(actor_grads)

    return total_loss, actor_loss, critic_loss, output_trainable_vars, output_grads 

  def apply_gradients(self, gradients, variables, optimizer):
    # list(...) is required for Python3.
    grads_and_vars = list(zip(gradients, variables))
    if self.gradient_clipping is not None:
      grads_and_vars = clip_gradient_norms(grads_and_vars,
                                           self.gradient_clipping)

    if self.summarize_grads_and_vars:
      add_variables_summaries(grads_and_vars, self.train_step_counter)
      add_variables_summaries(grads_and_vars, self.train_step_counter)

    optimizer.apply_gradients(grads_and_vars)

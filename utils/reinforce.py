from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import numpy as np
import tensorflow as tf

from beam_search.beam_search import  create_initial_beam_state_batch
from beam_search.beam_search import beam_search_step_batch
from beam_search.beam_search import BeamSearchConfig
from beam_search.beam_search import choose_top_k
from beam_search.beam_search import loc_optimal_beam_path_batch

from utils.tf_utils import batch_gather
from utils.tf_utils import soft_variables_update
from utils.tf_utils import Periodically
from utils.tf_utils import clip_gradient_norms
from utils.tf_utils import add_variables_summaries
from utils.tf_utils import tf_summary
from utils.tf_utils import bleu_score_tf
from utils.tf_utils import entropies
from utils.agent_utils import standard_normalize


#####################################
###### Reinforce Agent #############
#####################################

class SequenceBleuRewardFn(object):

  def __init__(self):
    pass

  def __call__(self, target_sequence, sample_sequence, **kwargs):
    # bleu_scores: [batch_size]
    return bleu_score_tf(target_sequence, sample_sequence)

class SequencePerStepRewardFn(object):

  def __init__(self, reward_fn):
    self.reward_fn = reward_fn

  def __call__(self,
               target_sequence,
               sample_sequence,
               **kwargs):
    rt_vec = []
    previous_reward = 0.0
    for time_step in range(1, sample_sequence.shape[1]):
      sub_seq = sample_sequence[:, 1:time_step+1]
      reward = self.reward_fn(target_sequence, sub_seq, **kwargs)
      rt = reward - previous_reward
      rt_vec.append(rt)
      previous_reward = reward
    return rt_vec

class ReinforceValueMlpNetwork(tf.keras.Model):
  
  def __init__(self, activation=None, dropout=0.0, name=''):
    super(ReinforceValueMlpNetwork, self).__init__()
    self.fc = tf.keras.layers.Dense(1,
                                    activation=activation,
                                    name='{}_fc'.format(name))
    self.dropout = tf.keras.layers.Dropout(dropout)

  def call(self, states, **kwargs):
    h_state = states[-1]
    state_value = self.fc(h_state)
    training = kwargs['training']
    state_value = self.dropout(state_value, training=training)
    state_value = tf.reshape(state_value, [-1])
    return state_value

class RnnDecoderNetwork(object):

   def __init__(self, rnn_decoder):
     self.rnn_decoder = rnn_decoder

   def __call__(self, inputs, **kwargs):
     predictions, state_next, _, _ = self.rnn_decoder(inputs, **kwargs)
     return predictions, state_next

   def reset_state(self, sequence, feat_d, **kwargs):
     return self.rnn_decoder.reset_state(sequence, feat_d, **kwargs)

   @property
   def variables(self):
     return self.rnn_decoder.variables

   @property
   def trainable_variables(self):
     return self.rnn_decoder.trainable_variables

   def get_weights(self):
     return self.rnn_decoder.get_weights()

   def set_weights(self, weights):
     return self.rnn_decoder.set_weights(weights)

   def create_variables(self):
     return self.rnn_decoder.create_variables()

   def clone(self):
     return type(self)(self.rnn_decoder.clone())

class ReinforceAgent(tf.keras.Model):
  # https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html
  def __init__(self,
               actor_network,
               sequence_reward_fn,
               src_feature_encoder,
               optimizer,
               sequence_sampler,
               start_token,
               eos_token,
               pad_token = 0,
               value_network = None,
               actor_loss_weight = 1.0,
               value_estimation_loss_coef = 0.2,
               advantage_fn = None,
               use_advantage_loss = True,
               prob_fn = tf.nn.softmax,
               log_prob_fn = tf.nn.log_softmax,
               gamma=1.0,
               normalize_returns=True,
               gradient_clipping=None,
               debug_summaries = True,
               summarize_grads_and_vars = False,
               entropy_regularization = None,
               train_step_counter = None,
               name = None):
      super(ReinforceAgent, self).__init__()

      self.actor_network = actor_network
      self.sequence_reward_fn = sequence_reward_fn
      self.value_network = value_network
      self.src_feature_encoder = src_feature_encoder
      self.sequence_sampler = sequence_sampler
      self.start_token = start_token
      self.eos_token = eos_token
      self.pad_token = pad_token

      self.optimizer = optimizer
      self.gamma = gamma
      self.normalize_returns = normalize_returns
      self.gradient_clipping = gradient_clipping
      self.entropy_regularization = entropy_regularization
      self.actor_loss_weight = actor_loss_weight
      self.value_estimation_loss_coef = value_estimation_loss_coef
      self.baseline = self.value_network is not None
      self.advantage_fn = advantage_fn
      if self.advantage_fn is None:
          if use_advantage_loss and self.baseline:
              self.advantage_fn =\
                  lambda returns, value_preds: returns - value_preds
          else:
              self.advantage_fn = lambda returns, _: returns

      self.prob_fn = prob_fn
      self.log_prob_fn = log_prob_fn
      self.debug_summaries = debug_summaries
      self.summarize_grads_and_vars = summarize_grads_and_vars
      self.train_step_counter = train_step_counter
      if train_step_counter is None:
          self.train_step_counter =\
              tf.compat.v1.train.get_or_create_global_step()

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

  def compute_reward_estimations(self, rewards, **kwargs):
    partial_rewards = tf.zeros((rewards[0].shape[0]), dtype=rewards[0].dtype)
    reward_estimations_vec = []
    for step in range(len(rewards) - 1, -1, -1):
      # r[step]: [batch_size]
      rewards_at_step = rewards[step]
      # G[step]: [batch_size]
      # G[step] = R[step] + gamma * G[step+1]
      partial_rewards = partial_rewards * self.gamma + rewards_at_step
      reward_estimations_vec.append(partial_rewards)
    return list(reversed(reward_estimations_vec)) 

  def entropy_loss(self, distributions, weights=None, dtype=tf.float32):
    """Computes entropy loss.

    Args:
      distributions: action distributions in shape of [batch_size, vocab_size]
      weights: Optional scalar or element-wise (per-batch-entry) importance
          weights.  Includes a mask for invalid timesteps.

    Returns:
      A Tensor representing the entropy loss.
    """

    with tf.name_scope('entropy_regularization'):
      entropy = tf.cast(entropies(distributions), dtype)
      if weights is not None:
        entropy *= weights
    return entropy

  def call(self, inputs, **kwargs):
    raw_features, sequences = inputs

    batch_size = sequences.shape[0]

    sequence_length = sequences.shape[1]

    policy_gradient_losses = 0.0
    entropy_losses = 0.0

    with tf.GradientTape() as tape:
      features = self.src_feature_encoder(raw_features, **kwargs)

      # Sample sequence Y' from X using delayed actor
      # with the same shape as sequences Y
      # sequence_samples: [batch_size, output_length]
      sequence_samples = self.sequence_sampler((features, sequences),
                                               **kwargs)
      sequence_samples = tf.stop_gradient(sequence_samples)

      # Compute reward at every timestep: r[t]
      # the reward of predicting the next token at timestep t+1
      # rewards: {[batch_size], [batch_size],...}: T-1

      rewards = self.sequence_reward_fn(sequences, sequence_samples, **kwargs)

      logging.debug("reinforce input sequences {} sequence_samples {} rewards {}".format(
         sequences, sequence_samples, rewards    
      ))

      dec_input_sample = tf.constant([self.start_token] * batch_size)

      # For Y', actor
      sample_sequence_state = self.actor_network.reset_state(sequence_samples,
                                                             {"features": features},
                                                             **kwargs)

      sample_log_prob_v = []
      advantages_v = []
      sequence_mask_v = []
      entropy_loss_v = []

      # estimate G[1:T]: [[batch_size]]
      # reward_estimations[t] => G[t+1]
      reward_estimations = self.compute_reward_estimations(rewards, **kwargs)
        
      for time_step in range(1, sequence_length):
        sample_predictions, sample_sequence_state_next = \
          self.actor_network((dec_input_sample,
                              {"features" : features},
                              sample_sequence_state),
                             **kwargs)
        sample_prob_dist = self.prob_fn(sample_predictions)
        # sample_log_prob_dist: [batch_size, vocab_size]
        # log(p(a|Y'[1:t-1], X))
        sample_log_prob_dist = self.log_prob_fn(sample_predictions)
        print("sample_predictions at step {}: {}".format(time_step, sample_predictions))
        print("sample_log_prob_dist at step {}: {}".format(time_step, sample_log_prob_dist))
        print("sample_prob_dist at step {}: {}".format(time_step, sample_prob_dist))

        dec_input_sample = sequence_samples[:, time_step]

        sequence_mask = tf.equal(dec_input_sample, self.pad_token)

        sequence_mask_v.append(sequence_mask[:, tf.newaxis])

        sample_exp = tf.expand_dims(dec_input_sample, -1)
        # sample_log_prob: [batch_size, 1]
        # log(p_theta(action[t] | state[1..t-1], X)) at time step t
        sample_log_prob = batch_gather(sample_log_prob_dist, sample_exp)
        print("sample_log_prob at step {}: {}".format(time_step, sample_log_prob))

        sample_log_prob_v.append(sample_log_prob)

        print("reward_estimations at step {}: {}".format(time_step, reward_estimations[time_step-1]))

        with tf.name_scope('Rewards'):
          tf_summary(
            name='policy_gradient_reward_estimations',
            tensor=reward_estimations,
            step=self.train_step_counter,
            mode='histogram')

        # obtain value estimation from state: V(s[t])
        value_preds = tf.zeros((batch_size), dtype=reward_estimations[time_step-1].dtype)
        if self.value_network is not None:
          value_preds = self.value_network(sample_sequence_state, **kwargs)
          value_preds = tf.reshape(value_preds, [-1])

       # advatnages: [batch_size]
        advantages = self.advantage_fn(reward_estimations[time_step-1], value_preds)

        print("advantages at step {}: {}".format(time_step, advantages))

        advantages_v.append(advantages[:, tf.newaxis])

        entropy_loss_step = self.entropy_loss(sample_prob_dist,
                                              weights=self.entropy_regularization,
                                              dtype=sample_prob_dist.dtype)
        entropy_loss_v.append(entropy_loss_step[:, tf.newaxis])

        sample_sequence_state = sample_sequence_state_next

      sample_log_probs = tf.concat(sample_log_prob_v, axis=-1) 
      sequence_masks = tf.concat(sequence_mask_v, axis=-1) 
      advantages_t = tf.concat(advantages_v, axis=-1)
      entropy_losses = tf.concat(entropy_loss_v, axis=-1)

      if self.normalize_returns:
        advantages_t = standard_normalize(advantages_t, axes=(0, 1))

      print("advantages total : {}".format(advantages_t))
      with tf.name_scope('Predictions'):
        tf_summary(
          name='policy_gradient_sample_log_probs',
          tensor=sample_log_probs,
          step=self.train_step_counter,
          mode='histogram')
        tf_summary(
          name='policy_gradient_advantages_t',
          tensor=advantages_t,
          step=self.train_step_counter,
          mode='histogram')
 
      policy_gradient_losses = -1 * sample_log_probs * advantages_t

      sequence_masks = tf.cast(sequence_masks, policy_gradient_losses.dtype)
      policy_gradient_losses *= sequence_masks

      sequence_masks = tf.cast(sequence_masks, entropy_losses.dtype)
      entropy_losses *= sequence_masks
      
      total_losses = policy_gradient_losses + entropy_losses

      tf.debugging.check_numerics(policy_gradient_losses,
                                  'Policy Gradient loss is inf or nan.')
      tf.debugging.check_numerics(entropy_losses,
                                  'Entropy loss is inf or nan.')

      total_loss = tf.reduce_mean(tf.reduce_sum(total_losses, -1))

      policy_gradient_tot_loss = tf.reduce_mean(tf.reduce_sum(policy_gradient_losses, -1))
      entropy_tot_loss = tf.reduce_mean(tf.reduce_sum(entropy_losses, -1))

      logging.debug("policy_gradient_losses {} entropy_losses {} total_losses {}".format(
        policy_gradient_losses, entropy_losses, total_losses
      ))

      print("policy_gradient_tot_loss {} entropy_tot_loss {} total_loss {}".format(
        policy_gradient_tot_loss, entropy_tot_loss, total_loss    
      ))

      trainable_variables = self.actor_network.trainable_variables + self.src_feature_encoder.trainable_variables

      if self.value_network is not None:
        trainable_variables += self.value_network.trainable_variables

      assert trainable_variables, (
          'No trainable variables to optimize.'
      )

      grads = tape.gradient(total_loss, trainable_variables)
      self.apply_gradients(grads, trainable_variables, self.optimizer)

      with tf.name_scope('Losses'):
        tf_summary(
          name='policy_gradient_losses',
          tensor=policy_gradient_losses,
          step=self.train_step_counter,
          mode='histogram')
        tf_summary(
          name='entropy_losses',
          tensor=entropy_losses,
          step=self.train_step_counter,
          mode='histogram')
        tf_summary(
          name='policy_gradient_tot_loss',
          tensor=policy_gradient_tot_loss,
          step=self.train_step_counter,
          mode='scalar')
        tf_summary(
          name='entropy_tot_loss',
          tensor=entropy_tot_loss,
          step=self.train_step_counter,
          mode='scalar')
        tf_summary(
          name='total_loss',
          tensor=total_loss,
          step=self.train_step_counter,
          mode='scalar')

    self.train_step_counter.assign_add(1)

    return total_loss, policy_gradient_tot_loss, trainable_variables, grads 

class SelfCriticalReinforceAgent(ReinforceAgent):
  # https://arxiv.org/pdf/1612.00563.pdf
  def __init__(self,
               actor_network,
               sequence_reward_fn,
               src_feature_encoder,
               optimizer,
               sequence_sampler,
               baseline_sequence_generator,
               start_token,
               eos_token,
               pad_token = 0,
               actor_loss_weight = 1.0,
               value_estimation_loss_coef = 0.2,
               advantage_fn = None,
               use_advantage_loss = True,
               prob_fn = tf.nn.softmax,
               log_prob_fn = tf.nn.log_softmax,
               gamma=1.0,
               normalize_returns=True,
               gradient_clipping=None,
               debug_summaries = True,
               summarize_grads_and_vars = False,
               entropy_regularization = None,
               train_step_counter = None,
               name = None):
    super(SelfCriticalReinforceAgent, self).__init__(
        actor_network,
        sequence_reward_fn,
        src_feature_encoder,
        optimizer,
        sequence_sampler,
        start_token,
        eos_token,
        pad_token=pad_token,
        value_network=None,
        actor_loss_weight=actor_loss_weight,
        value_estimation_loss_coef=value_estimation_loss_coef,
        advantage_fn=advantage_fn,
        use_advantage_loss=use_advantage_loss,
        prob_fn=prob_fn,
        log_prob_fn=log_prob_fn,
        gamma=gamma,
        normalize_returns=normalize_returns,
        gradient_clipping=gradient_clipping,
        debug_summaries=debug_summaries,
        summarize_grads_and_vars=summarize_grads_and_vars,
        entropy_regularization=entropy_regularization,
        train_step_counter=train_step_counter,
        name=name
    )
    self.baseline_sequence_generator = baseline_sequence_generator

  def call(self, inputs, **kwargs):
    raw_features, sequences = inputs

    batch_size = sequences.shape[0]

    sequence_length = sequences.shape[1]

    entropy_losses = 0.0

    with tf.GradientTape() as tape:
        features = self.src_feature_encoder(raw_features, **kwargs)

        # Sample sequence Y' from X using delayed actor
        # with the same shape as sequences Y
        # sequence_samples: [batch_size, output_length]
        sequence_samples = self.sequence_sampler((features, sequences),
                                                 **kwargs)

        sequence_samples = tf.stop_gradient(sequence_samples)
        print("sequence samples {}".format(sequence_samples))

        # Compute reward of the whole sequence sample
        # via comparing it against the golden target
        # rewards: [batch_size]

        rewards = self.sequence_reward_fn(sequences, sequence_samples,
                                          **kwargs)

        print("sample rewards {}".format(rewards))

        baseline_sequences = self.baseline_sequence_generator(features,
                                                              sequences,
                                                              **kwargs)
        baseline_sequences = tf.stop_gradient(baseline_sequences)

        print("baseline sequences {}".format(baseline_sequences))

        baseline_rewards = self.sequence_reward_fn(sequences,
                                                   baseline_sequences,
                                                   **kwargs)
        print("baseline rewards {}".format(baseline_rewards))

        logging.debug(
            "reinforce input sequences {} sequence_samples {} rewards {"
            "} baseline_sequences {} baseline_rewards {}".format(
                sequences, sequence_samples, rewards,
                baseline_sequences, baseline_rewards
            ))

        dec_input_sample = tf.constant([self.start_token] * batch_size)

        # For Y', actor
        sample_sequence_state = self.actor_network.reset_state(
            sequence_samples,
            {"features": features},
            **kwargs)

        sample_log_probs = 0.0

        for time_step in range(1, sequence_length):
          sample_predictions, sample_sequence_state_next = \
              self.actor_network((dec_input_sample,
                                  {"features": features},
                                  sample_sequence_state),
                                 **kwargs)
          sample_prob_dist = self.prob_fn(sample_predictions)
          # sample_log_prob_dist: [batch_size, vocab_size]
          # log(p(a|Y'[1:t-1], X))
          sample_log_prob_dist = self.log_prob_fn(sample_predictions)
          dec_input_sample = sequence_samples[:, time_step]

          sample_exp = tf.expand_dims(dec_input_sample, -1)
          # sample_log_prob: [batch_size]
          # log(p_theta(action[t] | state[1..t-1], X)) at time step t
          sample_log_prob = batch_gather(sample_log_prob_dist, sample_exp)
          sample_log_prob = tf.reshape(sample_log_prob, [-1])

          sequence_mask = tf.equal(dec_input_sample, self.pad_token)

          sequence_mask = tf.cast(sequence_mask, sample_log_prob.dtype)

          sample_log_prob *= sequence_mask

          sample_log_probs += sample_log_prob

          with tf.name_scope('Sample_predictions'):
            tf_summary(
                name='policy_gradient_sample_predictions',
                tensor=sample_predictions,
                step=self.train_step_counter,
                mode='histogram')

          entropy_loss_step = self.entropy_loss(sample_prob_dist,
                                                weights=self.entropy_regularization,
                                                dtype=sample_log_prob.dtype)
          entropy_loss_step *= sequence_mask
          entropy_losses += tf.reduce_mean(entropy_loss_step)

          sample_sequence_state = sample_sequence_state_next

        advantages_t = self.advantage_fn(rewards, 0.0)

        print("advantages_t {}".format(advantages_t))

        if self.normalize_returns:
          advantages_t = standard_normalize(advantages_t, axes=(0))

        with tf.name_scope('Predictions'):
          tf_summary(
              name='policy_gradient_sample_log_probs',
              tensor=sample_log_probs,
              step=self.train_step_counter,
              mode='histogram')
          tf_summary(
              name='policy_gradient_advantages_t',
              tensor=advantages_t,
              step=self.train_step_counter,
              mode='histogram')

        policy_gradient_losses = -1 * sample_log_probs * advantages_t

        print("policy_gradient_losses {}".format(policy_gradient_losses))
        print("entropy_losses {}".format(entropy_losses))

        policy_gradient_losses = tf.reduce_mean(policy_gradient_losses)

        total_losses = policy_gradient_losses + entropy_losses

        tf.debugging.check_numerics(policy_gradient_losses,
                                    'Policy Gradient loss is inf or nan.')
        tf.debugging.check_numerics(entropy_losses,
                                    'Entropy loss is inf or nan.')

        trainable_variables = self.actor_network.trainable_variables + \
                              self.src_feature_encoder.trainable_variables

        if self.value_network is not None:
            trainable_variables += self.value_network.trainable_variables

        assert trainable_variables, (
            'No trainable variables to optimize.'
        )

        grads = tape.gradient(total_losses, trainable_variables)
        self.apply_gradients(grads, trainable_variables, self.optimizer)

        with tf.name_scope('Losses'):
            tf_summary(
                name='policy_gradient_losses',
                tensor=policy_gradient_losses,
                step=self.train_step_counter,
                mode='histogram')
            tf_summary(
                name='entropy_losses',
                tensor=entropy_losses,
                step=self.train_step_counter,
                mode='histogram')
            tf_summary(
                name='total_losses',
                tensor=total_losses,
                step=self.train_step_counter,
                mode='scalar')

    self.train_step_counter.assign_add(1)

    return total_losses, policy_gradient_losses, trainable_variables, grads


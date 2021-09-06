import tensorflow as tf

from utils.tf_utils import batch_gather
from utils.tf_utils import _NEG_INF_FP16
from utils.tf_utils import _NEG_INF_FP32
from utils.tf_utils import tf_summary

class FlipThresholdDecayFunc(object):

    def __init__(self):
        pass

    def __call__(self, step, **kwargs):
        return self.call(step, **kwargs)

    def call(self, step, **kwargs):
        raise NotImplementedError

class InverseSigmoidDecay(FlipThresholdDecayFunc):

    def __init__(self, k):
        self.k = k
        assert self.k > 1

    def call(self, step, **kwargs):
        return self.k/(self.k + tf.exp(step/self.k))

class ExponentialDecay(FlipThresholdDecayFunc):

    def __init__(self, k):
        self.k = k
        assert self.k < 1

    def call(self, step, **kwargs):
        return tf.pow(self.k, step)

class LinearDecay(FlipThresholdDecayFunc):

    def __init__(self, gamma, k, c):
        self.gamma = gamma
        self.k = k
        self.c = c

    def call(self, step, **kwargs):
        return tf.maximum(self.gamma, self.k - self.c * step)

class ScheduledSampling(tf.keras.layers.Layer):

    def __init__(self, prob_thr_func, log_prob_fn = tf.nn.log_softmax):
      self.prob_thr_func = prob_thr_func
      self.log_prob_fn = log_prob_fn 
      super(ScheduledSampling, self).__init__()

    def call(self,
             target, # [batch_size, tgt_length]
             logits,  # [batch_size, vocab_size]
             step,
             summary_step,
             **kwargs):
      log_probs = self.log_prob_fn(logits)
      samples = tf.random.categorical(log_probs, 1)
      flip_threshold = self.prob_thr_func(step, **kwargs)
      choose_prob = tf.random.uniform((tf.shape(samples)[0], 1))
      samples = tf.cast(samples, target.dtype)
      choose_prob = tf.cast(choose_prob, tf.float64)
      flip_threshold = tf.cast(flip_threshold, tf.float64)
      tgt = target[:, step:step+1]
      tf_summary('sched_choose_prob', choose_prob, mode='histogram', step=summary_step)
      tf_summary('sched_flip_threshold', flip_threshold, mode='histogram', step=summary_step)
      return tf.where(tf.less(choose_prob, flip_threshold),
                      tgt,
                      samples)

class PredictProbConfidenceFunc(object):

    def __init__(self):
      pass

    def __call__(self, probs, tgt, **kwargs):
      # probs: [batch_size, vocab_size]
      # tgt: [batch_size, 1]

      # tgt_probs: [batch_size, 1]
      tgt_probs = batch_gather(probs, tgt)
      return tgt_probs

class ConfidenceScheduledSampling(tf.keras.layers.Layer):

    def __init__(self,
                 confidence_func,
                 gold_prob_thr_func,
                 rand_prob_thr_func,
                 log_prob_fn = tf.nn.log_softmax,
                 prob_fn = tf.nn.softmax):
      self.confidence_func = confidence_func
      self.gold_prob_thr_func = gold_prob_thr_func
      self.rand_prob_thr_func = rand_prob_thr_func
      self.log_prob_fn = log_prob_fn
      self.prob_fn = prob_fn
      super(ConfidenceScheduledSampling, self).__init__()

    def call(self,
             target, # [batch_size, length]
             logits,  # [batch_size, vocab_size]
             step,
             summary_step,
             **kwargs):
      log_probs = self.log_prob_fn(logits)

      # samples: [batch_size, 1] : y_gen
      samples = tf.random.categorical(log_probs, 1)  # y_gen
      samples = tf.cast(samples, target.dtype)

      # ground_truth: [batch_size, 1]: y 
      ground_truth = tf.expand_dims(target[:, step], -1)  # y_gold

      # random samle from target to produce y_rand
      # only sample from non_padding entries
      rand_num = tf.random.uniform(
          (tf.shape(target)[0], tf.shape(target)[1]))

      target_mask = tf.cast(target > 0, rand_num.dtype)

      neg_inf = _NEG_INF_FP16 if rand_num.dtype == tf.float16 else _NEG_INF_FP32

      rand_num += (1 - target_mask) * neg_inf

      # rand_indices: [batch_size, 1]
      _, rand_indices =  tf.nn.top_k(rand_num, 1)

      # rand_tgt: [batch_size, 1]
      rand_tgt = batch_gather(target, rand_indices)  # y_rand
      rand_tgt = tf.cast(rand_tgt, target.dtype)

      gold_prob = self.gold_prob_thr_func(step, **kwargs)
      rand_prob = self.rand_prob_thr_func(step, **kwargs)

      # confidence <= gold_prob
      #       select ground_truth
      # confidence <= rand_prob
      #       select sample
      # select rand_tgt

      tgt = target[:, step:step+1]
      probs = self.prob_fn(logits)
      # confidence: [batch_size, 1]
      confidence = self.confidence_func(probs, tgt, **kwargs)
      tf_summary('conf_sched_confidence', confidence, mode='histogram', step=summary_step)

      gold_prob = tf.cast(gold_prob, confidence.dtype)
      rand_prob = tf.cast(rand_prob, confidence.dtype)

      tf_summary('conf_sched_gold_prob', gold_prob, mode='scalar', step=summary_step)

      selected = tf.where(tf.less(confidence, gold_prob),
                   ground_truth,
                   tf.where(
                     tf.less(confidence, rand_prob),
                       samples,
                       rand_tgt
                     )
                   )

      return tf.cast(selected, target.dtype)

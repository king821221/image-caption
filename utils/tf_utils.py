# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Common TF utilities."""

import inspect
import numpy as np
import nltk
import six
import tensorflow as tf

from tensorflow.python.util import deprecation
from modeling import activations

_NEG_INF_FP16 = np.finfo(np.float16).min
_NEG_INF_FP32 = -1e9

def batch_gather(values, indices):
  a = values
  b = indices
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

def tf_summary(name, tensor, step, mode=''):
  if mode == 'scalar':
    tf.summary.scalar(name, tensor, step)
  elif mode == 'histogram':
    tf.summary.histogram(name, tensor, step)

def entropy(distribution, min_prob=1e-8):
  # p(x) * log(p(x))
  distribution = tf.maximum(distribution, min_prob)
  log_distrib = distribution * tf.math.log(distribution)
  reduction_axis = tf.range(tf.rank(log_distrib))[1:]
  return -tf.reduce_sum(log_distrib, axis=reduction_axis)

def entropies(distributions):
  """Computes total entropy of distribution.

  Args:
    distributions: A possibly batched tuple of distributions.
    action_spec: A nested tuple representing the action spec.

  Returns:
    A Tensor representing the entropy of each distribution in the batch.
    Assumes actions are independent, so that marginal entropies of each action
    may be summed.
  """

  entropies = [
    entropy(dist) for dist in tf.nest.flatten(distributions)
  ]

  # Sum entropies over action tuple.
  total_entropies = tf.add_n(entropies)

  return total_entropies

class Periodically(tf.Module):
  """Periodically performs the ops defined in `body`."""

  def __init__(self, body, period, name='periodically'):
    """Periodically performs the ops defined in `body`.

    The body tensorflow op will be executed every `period` times the
    periodically op is executed. More specifically, with `n` the number of times
    the op has been executed, the body will be executed when `n` is a non zero
    positive multiple of `period` (i.e. there exist an integer `k > 0` such that
    `k * period == n`).

    If `period` is `None`, it will not perform any op and will return a
    `tf.no_op()`.

    If `period` is 1, it will just execute the body, and not create any counters
    or conditionals.

    Args:
      body: callable that returns the tensorflow op to be performed every time
        an internal counter is divisible by the period. The op must have no
        output (for example, a tf.group()).
      period: inverse frequency with which to perform the op. It can be a Tensor
        or a Variable.
      name: name of the object.

    Raises:
      TypeError: if body is not a callable.

    Returns:
      An op that periodically performs the specified op.
    """
    super(Periodically, self).__init__(name=name)
    if not callable(body):
      raise TypeError('body must be callable.')
    self._body = body
    self._period = period
    self._counter = tf.compat.v1.get_variable(self.name + '/counter', 0)

  def __call__(self):

    def call(strategy=None):
      del strategy  # unused
      if self._period is None:
        return tf.no_op()
      if self._period == 1:
        return self._body()
      period = tf.cast(self._period, self._counter.dtype)
      remainder = tf.math.mod(self._counter.assign_add(1), period)
      return tf.cond(
          pred=tf.equal(remainder, 0), true_fn=self._body, false_fn=tf.no_op)

    # TODO(b/129083817) add an explicit unit test to ensure correct behavior
    ctx = tf.distribute.get_replica_context()
    if ctx:
      return tf.distribute.get_replica_context().merge_call(call)
    else:
      return call()

def soft_variables_update(source_variables,
                          target_variables,
                          tau=1.0,
                          tau_non_trainable=None,
                          sort_variables_by_name=False):
  """Performs a soft/hard update of variables from the source to the target.

  For each variable v_t in target variables and its corresponding variable v_s
  in source variables, a soft update is:
  v_t = (1 - tau) * v_t + tau * v_s

  When tau is 1.0 (the default), then it does a hard update:
  v_t = v_s

  Args:
    source_variables: list of source variables.
    target_variables: list of target variables.
    tau: A float scalar in [0, 1]. When tau is 1.0 (the default), we do a hard
      update. This is used for trainable variables.
    tau_non_trainable: A float scalar in [0, 1] for non_trainable variables. If
      None, will copy from tau.
    sort_variables_by_name: A bool, when True would sort the variables by name
      before doing the update.

  Returns:
    An operation that updates target variables from source variables.
  Raises:
    ValueError: if `tau not in [0, 1]`.
    ValueError: if `len(source_variables) != len(target_variables)`.
  """
  if tau < 0 or tau > 1:
    raise ValueError('Input `tau` should be in [0, 1].')
  if tau_non_trainable is None:
    tau_non_trainable = tau

  if tau_non_trainable < 0 or tau_non_trainable > 1:
    raise ValueError('Input `tau_non_trainable` should be in [0, 1].')

  updates = []

  op_name = 'soft_variables_update'
  if tau == 0.0 or not source_variables or not target_variables:
    return tf.no_op(name=op_name)
  if len(source_variables) != len(target_variables):
    raise ValueError(
        'Source and target variable lists have different lengths: '
        '{} vs. {}'.format(len(source_variables), len(target_variables)))
  if sort_variables_by_name:
    source_variables = sorted(source_variables, key=lambda x: x.name)
    target_variables = sorted(target_variables, key=lambda x: x.name)

  strategy = tf.distribute.get_strategy()

  for (v_s, v_t) in zip(source_variables, target_variables):
    v_t.shape.assert_is_compatible_with(v_s.shape)

    def update_fn(v1, v2):
      """Update variables."""
      # For not trainable variables do hard updates.
      # This helps stabilaze BatchNorm moving averagees TODO(b/144455039)
      if not v1.trainable:
        current_tau = tau_non_trainable
      else:
        current_tau = tau

      if current_tau == 1.0:
        return v1.assign(v2)
      else:
        print("update v1 {} v2 {}".format(v1.name, v2.name))
        return v1.assign((1 - current_tau) * v1 + current_tau * v2)

    # TODO(b/142508640): remove this when b/142802462 is fixed.
    # Workaround for b/142508640, only use extended.update for
    # MirroredVariable variables (which are trainable variables).
    # For other types of variables (i.e. SyncOnReadVariables, for example
    # batch norm stats) do a regular assign, which will cause a sync and
    # broadcast from replica 0, so will have slower performance but will be
    # correct and not cause a failure.
    if tf.distribute.has_strategy() and v_t.trainable:
      # Assignment happens independently on each replica,
      # see b/140690837 #46.
      update = strategy.extended.update(v_t, update_fn, args=(v_s,))
    else:
      update = update_fn(v_t, v_s)

    updates.append(update)
  return tf.group(*updates, name=op_name)

def clip_gradient_norms(gradients_to_variables, max_norm):
  """Clips the gradients by the given value.

  Args:
    gradients_to_variables: A list of gradient to variable pairs (tuples).
    max_norm: the maximum norm value.

  Returns:
    A list of clipped gradient to variable pairs.
  """
  clipped_grads_and_vars = []
  for grad, var in gradients_to_variables:
    if grad is not None:
      if isinstance(grad, tf.IndexedSlices):
        tmp = tf.clip_by_norm(grad.values, max_norm)
        grad = tf.IndexedSlices(tmp, grad.indices, grad.dense_shape)
      else:
        grad = tf.clip_by_norm(grad, max_norm)
    clipped_grads_and_vars.append((grad, var))
  return clipped_grads_and_vars

def add_variables_summaries(grads_and_vars, step):
  """Add summaries for variables.

  Args:
    grads_and_vars: A list of (gradient, variable) pairs.
    step: Variable to use for summaries.
  """
  with tf.name_scope('summarize_vars'):
    for _, var in grads_and_vars:
      if isinstance(var, tf.IndexedSlices):
        var_values = var.values
      else:
        var_values = var
      var_name = var.name.replace(':', '_')
      tf.compat.v2.summary.histogram(
          name=var_name + '_value', data=var_values, step=step)
      tf.compat.v2.summary.scalar(
          name=var_name + '_value_norm',
          data=tf.linalg.global_norm([var_values]),
          step=step)

def tf_print(tensor, message, summarize=100, level=0):
  if level>0:
    tensor = tf.Print(tensor,
                      [tf.shape(tensor),
                       tf.reduce_min(tensor),
                       tf.reduce_max(tensor),
                       tf.reduce_mean(tensor),
                       tensor],
                      message=message,
                      summarize=summarize)
  return tensor

eos_token = 4
start_token = 3

def bleu_score_py(real, predict):
  bleu_vec = []

  for idx in range(len(real)):
      rl = list(real[idx])
      pr = list(predict[idx])

      new_rl = []

      for e in rl:
        if e == eos_token:
          new_rl.append(e)
          break
        if e != start_token:
          new_rl.append(e)

      new_pr = []

      for e in pr:
        if e == eos_token:
          new_rl.append(e)
          break
        if e != start_token:
          new_pr.append(e)

      bleu = nltk.translate.bleu_score.sentence_bleu([new_rl], new_pr)
      bleu_vec.append(bleu)

  return np.array(bleu_vec).astype(np.float32)

def bleu_score_tf(real, pred):
  y = tf.numpy_function(bleu_score_py, [real, pred], tf.float32)
  return y

@deprecation.deprecated(
    None,
    "tf.keras.layers.Layer supports multiple positional args and kwargs as "
    "input tensors. pack/unpack inputs to override __call__ is no longer "
    "needed.")
def pack_inputs(inputs):
  """Pack a list of `inputs` tensors to a tuple.

  Args:
    inputs: a list of tensors.

  Returns:
    a tuple of tensors. if any input is None, replace it with a special constant
    tensor.
  """
  inputs = tf.nest.flatten(inputs)
  outputs = []
  for x in inputs:
    if x is None:
      outputs.append(tf.constant(0, shape=[], dtype=tf.int32))
    else:
      outputs.append(x)
  return tuple(outputs)


@deprecation.deprecated(
    None,
    "tf.keras.layers.Layer supports multiple positional args and kwargs as "
    "input tensors. pack/unpack inputs to override __call__ is no longer "
    "needed.")
def unpack_inputs(inputs):
  """unpack a tuple of `inputs` tensors to a tuple.

  Args:
    inputs: a list of tensors.

  Returns:
    a tuple of tensors. if any input is a special constant tensor, replace it
    with None.
  """
  inputs = tf.nest.flatten(inputs)
  outputs = []
  for x in inputs:
    if is_special_none_tensor(x):
      outputs.append(None)
    else:
      outputs.append(x)
  x = tuple(outputs)

  # To trick the very pointless 'unbalanced-tuple-unpacking' pylint check
  # from triggering.
  if len(x) == 1:
    return x[0]
  return tuple(outputs)


def is_special_none_tensor(tensor):
  """Checks if a tensor is a special None Tensor."""
  return tensor.shape.ndims == 0 and tensor.dtype == tf.int32


def get_activation(identifier, use_keras_layer=False):
  """Maps a identifier to a Python function, e.g., "relu" => `tf.nn.relu`.

  It checks string first and if it is one of customized activation not in TF,
  the corresponding activation will be returned. For non-customized activation
  names and callable identifiers, always fallback to tf.keras.activations.get.

  Prefers using keras layers when use_keras_layer=True. Now it only supports
  'relu', 'linear', 'identity', 'swish'.

  Args:
    identifier: String name of the activation function or callable.
    use_keras_layer: If True, use keras layer if identifier is allow-listed.

  Returns:
    A Python function corresponding to the activation function or a keras
    activation layer when use_keras_layer=True.
  """
  if isinstance(identifier, six.string_types):
    identifier = str(identifier).lower()
    if use_keras_layer:
      keras_layer_allowlist = {
          "relu": "relu",
          "linear": "linear",
          "identity": "linear",
          "swish": "swish",
          "sigmoid": "sigmoid",
          "relu6": tf.nn.relu6,
      }
      if identifier in keras_layer_allowlist:
        return tf.keras.layers.Activation(keras_layer_allowlist[identifier])
    name_to_fn = {
        "gelu": activations.gelu,
        "simple_swish": activations.simple_swish,
        "hard_swish": activations.hard_swish,
        "relu6": activations.relu6,
        "hard_sigmoid": activations.hard_sigmoid,
        "identity": activations.identity,
    }
    if identifier in name_to_fn:
      return tf.keras.activations.get(name_to_fn[identifier])
  return tf.keras.activations.get(identifier)


def get_shape_list(tensor, expected_rank=None, name=None):
  """Returns a list of the shape of tensor, preferring static dimensions.

  Args:
    tensor: A tf.Tensor object to find the shape of.
    expected_rank: (optional) int. The expected rank of `tensor`. If this is
      specified and the `tensor` has a different rank, and exception will be
      thrown.
    name: Optional name of the tensor for the error message.

  Returns:
    A list of dimensions of the shape of tensor. All static dimensions will
    be returned as python integers, and dynamic dimensions will be returned
    as tf.Tensor scalars.
  """
  if expected_rank is not None:
    assert_rank(tensor, expected_rank, name)

  shape = tensor.shape.as_list()

  non_static_indexes = []
  for (index, dim) in enumerate(shape):
    if dim is None:
      non_static_indexes.append(index)

  if not non_static_indexes:
    return shape

  dyn_shape = tf.shape(tensor)
  for index in non_static_indexes:
    shape[index] = dyn_shape[index]
  return shape


def assert_rank(tensor, expected_rank, name=None):
  """Raises an exception if the tensor rank is not of the expected rank.

  Args:
    tensor: A tf.Tensor to check the rank of.
    expected_rank: Python integer or list of integers, expected rank.
    name: Optional name of the tensor for the error message.

  Raises:
    ValueError: If the expected shape doesn't match the actual shape.
  """
  expected_rank_dict = {}
  if isinstance(expected_rank, six.integer_types):
    expected_rank_dict[expected_rank] = True
  else:
    for x in expected_rank:
      expected_rank_dict[x] = True

  actual_rank = tensor.shape.ndims
  if actual_rank not in expected_rank_dict:
    raise ValueError(
        "For the tensor `%s`, the actual tensor rank `%d` (shape = %s) is not "
        "equal to the expected tensor rank `%s`" %
        (name, actual_rank, str(tensor.shape), str(expected_rank)))


def safe_mean(losses):
  """Computes a safe mean of the losses.

  Args:
    losses: `Tensor` whose elements contain individual loss measurements.

  Returns:
    A scalar representing the mean of `losses`. If `num_present` is zero,
      then zero is returned.
  """
  total = tf.reduce_sum(losses)
  num_elements = tf.cast(tf.size(losses), dtype=losses.dtype)
  return tf.math.divide_no_nan(total, num_elements)

def get_kwargs():
  frame = inspect.currentframe().f_back
  keys, _, _, values = inspect.getargvalues(frame)
  kwargs = {}
  for key in keys:
    if key != 'self':
      kwargs[key] = values[key]
  return kwargs


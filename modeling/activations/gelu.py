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

"""Gaussian error linear unit."""

import tensorflow as tf
from tensorflow.python.ops import nn
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops

def gelu_inner(features, approximate=False, name=None):
  """Compute the Gaussian Error Linear Unit (GELU) activation function.
  Gaussian error linear unit (GELU) computes
  `x * P(X <= x)`, where `P(X) ~ N(0, 1)`.
  The (GELU) nonlinearity weights inputs by their value, rather than gates
  inputs by their sign as in ReLU.
  For example:
  >>> x = tf.constant([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=tf.float32)
  >>> y = tf.nn.gelu(x)
  >>> y.numpy()
  array([-0.00404951, -0.15865529,  0.        ,  0.8413447 ,  2.9959507 ],
      dtype=float32)
  >>> y = tf.nn.gelu(x, approximate=True)
  >>> y.numpy()
  array([-0.00363752, -0.15880796,  0.        ,  0.841192  ,  2.9963627 ],
      dtype=float32)
  Args:
    features: A `Tensor` representing preactivation values.
    approximate: An optional `bool`. Defaults to `False`. Whether to enable
      approximation.
    name: A name for the operation (optional).
  Returns:
    A `Tensor` with the same type as `features`.
  References:
    [Gaussian Error Linear Units (GELUs)](https://arxiv.org/abs/1606.08415).
  """
  with ops.name_scope(name, "Gelu", [features]):
    features = ops.convert_to_tensor(features, name="features")
    if approximate:
      coeff = math_ops.cast(0.044715, features.dtype)
      return 0.5 * features * (
          1.0 + math_ops.tanh(0.7978845608028654 *
                              (features + coeff * math_ops.pow(features, 3))))
    else:
      return 0.5 * features * (1.0 + math_ops.erf(
          features / math_ops.cast(1.4142135623730951, features.dtype)))

@tf.keras.utils.register_keras_serializable(package='Text')
def gelu(x):
  """Gaussian Error Linear Unit.

  This is a smoother version of the RELU.
  Original paper: https://arxiv.org/abs/1606.08415
  Args:
    x: float Tensor to perform activation.

  Returns:
    `x` with the GELU activation applied.
  """
  return gelu_inner(x, approximate=True)

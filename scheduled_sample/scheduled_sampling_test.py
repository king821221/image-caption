# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Test Scheduled Sampling methods."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

from scheduled_sample.scheduled_sampling import ScheduledSampling
from scheduled_sample.scheduled_sampling import InverseSigmoidDecay

tf.random.set_seed(time.time())

class ScheduledSamplingTest(tf.test.TestCase):

  def test_sched_sample(self):
    for i in range(5):
      ground_truth = tf.constant([[2], [1], [0]])
      logits = tf.constant([[0.3,0.1,0.2], [0.01,0.05,0.02], [0.001,0.002,0.003]])
      log_probs = tf.nn.log_softmax(logits)
      time = tf.constant(i)
      sched_sampling = ScheduledSampling(InverseSigmoidDecay(5))
      next = sched_sampling(ground_truth, log_probs, time)
      print("NEXT sample {}".format(next))


if __name__ == "__main__":
  tf.test.main()


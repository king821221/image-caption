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
"""Test Reward methods."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from utils.reinforce import SequenceBleuRewardFn 
from utils.reinforce import SequencePerStepRewardFn 


class RewardFnTest(tf.test.TestCase):

  def test_reward_fn(self):
    sequence = [[3,2,53,42,101,23,14,2,16,43,4,0,0,0,0,0,0], [3,2,38,79,247,248,2,38,8,80,56,9,249,250,251,252,4]] 
    sequence_samples = [[3,4060,2895,1596,2851,2292,2201,3020,2149,658,4,0,0,0,0,0,0], [3,4940,1115,688,4928,2181,3130,3468,3184,2366,1616,136,393,4383,4288,1488,4]]

    sequence = np.array(sequence)
    sequence_samples = np.array(sequence_samples)

    sequence = tf.convert_to_tensor(sequence, dtype=tf.int32)
    sequence_samples = tf.convert_to_tensor(sequence_samples, dtype=tf.int32)

    reward_fn = SequencePerStepRewardFn(SequenceBleuRewardFn())

    rewards = reward_fn(sequence, sequence_samples)

    print(rewards)

if __name__ == "__main__":
  tf.test.main()

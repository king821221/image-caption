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
"""Test Beam Search methods."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from beam_search import create_initial_beam_state
from beam_search import create_initial_beam_state_batch
from beam_search import BeamSearchConfig
from beam_search import choose_top_k
from beam_search import beam_search_step
from beam_search import beam_search_step_batch
from beam_search import loc_optimal_beam_path
from beam_search import loc_optimal_beam_path_batch


class BeamSearchTest(tf.test.TestCase):

  def test_single_batch(self):
    vocab_size = 5
    time = tf.constant(0)
    beam_width = 2
    beam_state = create_initial_beam_state(beam_width)
    beam_sequence = []
    beam_config = BeamSearchConfig(beam_width=beam_width,
                                   vocab_size=vocab_size,
                                   eos_token=0,
                                   length_penalty_weight=1.2,
                                   choose_successors_fn=choose_top_k)
    logits = tf.constant([[-0.1, -0.2, -0.3, -0.4, -0.5], [-0.01, -0.02, -0.03, -0.04, -0.05]], dtype=tf.float32)
    beam_out, next_beam_state = beam_search_step(time, logits, beam_state, beam_config)
    beam_sequence.append((beam_out, next_beam_state))
    print("step 1 beam out {} beam_out_state {}".format(beam_out, next_beam_state))
    logits = tf.constant([[1.01, 0.4, 0.3, 0.2, 0.1], [0.01, 1.5, 0.03, 0.04, 0.05]], dtype=tf.float32)
    beam_out, next_beam_state = beam_search_step(time+1, logits, next_beam_state, beam_config)
    print("step 2 beam out {} beam_out_state {}".format(beam_out, next_beam_state))
    beam_sequence.append((beam_out, next_beam_state))
    decode_sequence = loc_optimal_beam_path(beam_sequence)
    result = []
    for e in decode_sequence:
      result.append(e.numpy())
    self.assertAllEqual(result,  [1,0])

  def test_multiple_batch(self):
    vocab_size = 5
    time = tf.constant(0)
    batch_size = 2
    beam_width = 2
    beam_state = create_initial_beam_state_batch(batch_size, beam_width)
    beam_sequence = []
    beam_config = BeamSearchConfig(beam_width=beam_width,
                                   vocab_size=vocab_size,
                                   eos_token=0,
                                   length_penalty_weight=1.2,
                                   choose_successors_fn=choose_top_k)
    # logits: [batch_size, beam_width, vocab_size]
    logits = tf.constant([
      [
        [-0.099, -0.212, -.003, -0.4, -0.5],
        [0.02, -0.01, 0.03, 0.15, -0.05]
      ],
      [
        [-0.1, -0.2, -0.3, -0.4, -0.5],
        [-0.01, -0.02, -0.03, -0.04, -0.05]
      ]
    ], dtype=tf.float32)
    beam_out, next_beam_state = beam_search_step_batch(time, logits, beam_state, beam_config)
    beam_sequence.append((beam_out, next_beam_state))
    print("step 1 beam out {} beam_out_state {}".format(beam_out, next_beam_state))

    logits = tf.constant([
      [
        [0.59, -0.21, 1.3, 2.5, -0.5],
        [0.01, 0.05, 0.03, -0.05, 1.05]
      ],
      [
        [1.01, 0.4, 0.3, 0.2, 0.1],
        [0.01, 1.5, 0.03, 0.04, 0.05]
      ]
    ], dtype=tf.float32)

    beam_out, next_beam_state = beam_search_step_batch(time+1, logits, next_beam_state, beam_config)
    print("step 2 beam out {} beam_out_state {}".format(beam_out, next_beam_state))
    beam_sequence.append((beam_out, next_beam_state))

    decode_sequence = loc_optimal_beam_path_batch(beam_sequence)
    result = []
    for e in decode_sequence:
      result.append(np.reshape(e.numpy(), (-1)))
    self.assertAllEqual(result[0],  [2,1])
    self.assertAllEqual(result[1],  [3,0])

if __name__ == "__main__":
  tf.test.main()


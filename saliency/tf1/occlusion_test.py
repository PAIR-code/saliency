# Copyright 2021 Google Inc. All Rights Reserved.
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

"""Tests accuracy and correct TF1 usage for occlusion."""
import unittest
import unittest.mock as mock

from . import occlusion
from . import utils
import numpy as np
import tensorflow.compat.v1 as tf


class OcclusionTest(unittest.TestCase):
  """To run: "python -m saliency.tf1.occlusion_test" top-level saliency directory."""

  def setUp(self):
    super().setUp()
    self.max_sigma = 10
    with tf.Graph().as_default() as graph:
      self.x = tf.placeholder(shape=[None, 5, 5], dtype=tf.float32)
      # Define function to just multiply all values by 3
      y = self.x[:, 1, 1] * 3 + self.x[:, 3, 3] * 3
      self.sess = tf.Session(graph=graph)
      self.sess_spy = mock.MagicMock(wraps=self.sess)
      # All black except white at [1,1] and [3,3].
      self.x_input_val = np.array([
          [0.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 1.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 1.0, 0.0],
          [0.0, 0.0, 0.0, 0.0, 0.0],
      ], dtype=float)

      # The computed Occlusion mask should bias the center, but the quadrants
      # with nonzero values should be greater than the other two (exact expected
      # values stored in ref_mask).
      self.expected_val = np.array([[3., 6., 6., 3., 0.],
                                    [6., 15., 18., 12., 3.],
                                    [6., 18., 24., 18., 6.],
                                    [3., 12., 18., 15., 6.],
                                    [0., 3., 6., 6., 3.]])
      self.occlusion_instance = occlusion.Occlusion(graph, self.sess_spy, y,
                                                    self.x)

  def testOcclsuionGetMask(self):
    """Tests the Occlusion method using a simple TF1 model."""
    expected_calls = 10  # 5x5 image with size 3 window, 1 extra call for input
    # Calculate the occlusion attribution of the input.
    mask = self.occlusion_instance.GetMask(self.x_input_val,
                                           feed_dict=None,
                                           size=3,
                                           value=0)

    # Verify the result for completeness..
    # Expected (for max_sigma=10): 1.4746742
    # mask.sum (for max_sigma=10): 1.4742470...
    np.testing.assert_equal(mask, self.expected_val)
    self.assertEqual(self.sess_spy.run.call_count, expected_calls)

  def testOcclusionGetMaskArgs(self):
    """Tests the sess.run receives the correct inputs."""
    feed_dict = {'foo': 'bar'}
    self.sess_spy.run.return_value = [[3]]

    self.occlusion_instance.GetMask(
        self.x_input_val, feed_dict=feed_dict, size=3, value=0)
    actual_feed_dict = self.sess_spy.run.call_args[1]['feed_dict']

    self.assertEqual(actual_feed_dict['foo'], feed_dict['foo'])

if __name__ == '__main__':
  unittest.main()

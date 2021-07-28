# Copyright 2021 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests completeness axiom, batching, and error handling for blur_ig."""
import unittest
import unittest.mock as mock

from . import blur_ig
import numpy as np
from scipy.ndimage import gaussian_filter
import tensorflow.compat.v1 as tf


class BlurIgTest(unittest.TestCase):
  """To run: "python -m saliency.tf1.blur_ig_test" top-level saliency directory."""

  def setUp(self):
    super().setUp()
    self.max_sigma = 10
    with tf.Graph().as_default() as graph:
      self.x = tf.placeholder(shape=[None, 5, 5, 1], dtype=tf.float32)
      y = tf.sin(self.x)
      y_sum = tf.reduce_sum(y, [1,2,3])
      self.gradients_node = tf.gradients(y, self.x)[0]
      self.sess = tf.Session(graph=graph)
      self.sess_spy = mock.MagicMock(wraps=self.sess)
    # All black except 2 pixels near the center.
      self.x_input_val = np.array([
          [0.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 0.5, 0.0, 0.0, 0.0],
          [0.0, 0.0, 1.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0, 0.0],
      ],
                                  dtype=float)
      self.x_input_val = self.x_input_val.reshape((5, 5, 1))
      # Calculate the value of `y` at the input.
      y_input_val = self.sess.run(y, feed_dict={self.x: [self.x_input_val]})

      # Baseline is the fully blurred version of the input.
      x_baseline_val = gaussian_filter(
          self.x_input_val,
          sigma=[self.max_sigma, self.max_sigma, 0],
          mode='constant')
      y_baseline_val = self.sess.run(y, feed_dict={self.x: [x_baseline_val]})

      # The expected BlurIG value is equal to the difference between
      # the `y` value at the input and the `y` value at the baseline. Because
      # each value is independent, we can calculate the expected blur_ig value
      # of each.
      #
      # Expected: [[-0, -0, -0, -0, -0],
      #            [-0, 0.641, -0, -0, -0],
      #            [-0, -0, 0.838, -0, -0],
      #            [-0, -0, -0, -0, -0],
      #            [-0, -0, -0, -0, -0]
      self.expected_val = y_input_val[0] - y_baseline_val[0]
      self.blur_ig_instance = blur_ig.BlurIG(graph,
                                             self.sess_spy,
                                             y_sum,
                                             self.x)

  def testBlurIGGetMask(self):
    """Tests that BlurIG steps are created and aggregated correctly."""
    x_steps = 2000

    # Calculate the Blur IG attribution of the input.
    mask = self.blur_ig_instance.GetMask(self.x_input_val,
                                         feed_dict=None,
                                         max_sigma=self.max_sigma,
                                         steps=x_steps)

    # Because the baseline is blurred, all zero values should still have some
    # attribution (introduced noise).
    self.assertEqual(np.count_nonzero(mask), mask.size)
    # Verify the result (for accuracy and therefore completeness).
    np.testing.assert_almost_equal(mask, self.expected_val, decimal=2)
    self.assertEqual(self.sess_spy.run.call_count, x_steps)

  def testBlurIGGetMaskBatched(self):
    """Tests that multiple BlurIG batches are created and aggregated correctly."""
    x_steps = 1001
    batch_size = 500
    expected_calls = 3  # batch size is 500, ceil(1001/500)=3
    self.blur_ig_instance.validate_xy_tensor_shape = mock.MagicMock()
    expected_validate_args = (x_steps, batch_size)

    mask = self.blur_ig_instance.GetMask(self.x_input_val,
                                         feed_dict=None,
                                         max_sigma=self.max_sigma,
                                         steps=x_steps,
                                         batch_size=batch_size)
    validate_args = self.blur_ig_instance.validate_xy_tensor_shape.call_args[0]

    # Because the baseline is blurred, all zero values should still have some
    # attribution (introduced noise).
    self.assertEqual(np.count_nonzero(mask), mask.size)
    # Verify the result (for accuracy and therefore completeness).
    np.testing.assert_almost_equal(mask, self.expected_val, decimal=2)
    self.assertEqual(self.sess_spy.run.call_count, expected_calls)
    self.assertEqual(validate_args, expected_validate_args)

  def testBlurIGGetMaskSingleBatch(self):
    """Tests that a single BlurIG batch is created and aggregated correctly."""
    x_steps = 999
    batch_size = 1000
    expected_calls = 1  # batch size is 1000, ceil(999/1000)=1

    mask = self.blur_ig_instance.GetMask(self.x_input_val,
                                         feed_dict=None,
                                         max_sigma=self.max_sigma,
                                         steps=x_steps,
                                         batch_size=batch_size)

    # Because the baseline is blurred, all zero values should still have some
    # attribution (introduced noise).
    self.assertEqual(np.count_nonzero(mask), mask.size)
    # Verify the result (for accuracy and therefore completeness).
    np.testing.assert_almost_equal(mask, self.expected_val, decimal=2)
    self.assertEqual(self.sess_spy.run.call_count, expected_calls)

  def testBlurIGGetMaskArgs(self):
    """Tests that call_model_function receives correct inputs."""
    x_steps = 5
    feed_dict = {'foo': 'bar'}
    self.sess_spy.run.return_value = [self.x_input_val.reshape((1, 5, 5, 1))]

    self.blur_ig_instance.GetMask(self.x_input_val,
                                  feed_dict=feed_dict,
                                  max_sigma=self.max_sigma,
                                  steps=x_steps)
    actual_feed_dict = self.sess_spy.run.call_args[1]['feed_dict']

    self.assertEqual(actual_feed_dict['foo'], feed_dict['foo'])

if __name__ == '__main__':
  unittest.main()

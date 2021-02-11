# Copyright 2020 Google Inc. All Rights Reserved.
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

"""Tests completeness axiom, batching, and error handling for blur_ig.
"""
import unittest

from . import blur_ig
import numpy as np
import tensorflow.compat.v1 as tf
from scipy.ndimage import gaussian_filter


class BlurIgTest(unittest.TestCase):
  """To run: "python -m saliency.blur_ig_test" top-level saliency directory."""

  def setUp(self):
    super().setUp()
    self.max_sigma = 10
    with tf.Graph().as_default() as graph:
      self.x = tf.placeholder(shape=[None, 5, 5, 1], dtype=tf.float32)
      # Define function to just look at center pixel.
      y = tf.sin(self.x)
      self.gradients_node = tf.gradients(y, self.x)[0]
      self.sess = tf.Session(graph=graph)
      self.sess_spy = unittest.mock.MagicMock(wraps=self.sess)
      # All black except 1 white pixel at the center.
      self.x_input_val = np.array([
          [0.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 0.5, 0.0, 0.0, 0.0],
          [0.0, 0.0, 1.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0, 0.0],
      ],
                                  dtype=np.float)
      self.x_input_val = self.x_input_val.reshape((5, 5, 1))
      # Calculate the value of `y` at the input.
      y_input_val = self.sess.run(y, feed_dict={self.x: [self.x_input_val]})

      # Baseline is the fully blurred version of the input.
      x_baseline_val = gaussian_filter(self.x_input_val,
                          sigma=[self.max_sigma, self.max_sigma, 0],
                          mode='constant')
      y_baseline_val = self.sess.run(y, feed_dict={self.x: [x_baseline_val]})

      # Test if completeness axiom is satisfied.
      # The expected BlurIG value is equal to the difference between
      # the `y` value at the input and the `y` value at the baseline.
      self.expected_val = y_input_val[0] - y_baseline_val[0]
      self.blur_ig_instance = blur_ig.BlurIG(graph,
                                        self.sess_spy,
                                        y,
                                        self.x)

  def testBlurIGGetMask(self):
    x_steps = 2000

    # Calculate the Blur IG attribution of the input.
    mask = self.blur_ig_instance.GetMask(self.x_input_val, 
                                         feed_dict={}, 
                                         max_sigma=self.max_sigma, 
                                         steps=x_steps)

    # Verify the result for completeness..
    # Expected (for max_sigma=10): 1.4746742
    # mask.sum (for max_sigma=10): 1.4742470...
    np.testing.assert_almost_equal(mask, self.expected_val, decimal=2)
    self.assertEqual(self.sess_spy.run.call_count, x_steps)

  def testBlurIGGetMaskBatched(self):
    x_steps = 1001
    batch_size = 500
    expected_calls = 3  # batch size is 500, ceil(1001/500)=3

    mask = self.blur_ig_instance.GetMask(self.x_input_val, 
                                         feed_dict={}, 
                                         max_sigma=self.max_sigma, 
                                         steps=x_steps,
                                         batch_size=batch_size)

    # Verify the result.
    np.testing.assert_almost_equal(mask, self.expected_val, decimal=2)
    self.assertEqual(self.sess_spy.run.call_count, expected_calls)

  def testBlurIGGetMaskSingleBatch(self):
    x_steps = 999
    batch_size = 1000
    expected_calls = 1  # batch size is 1000, ceil(999/1000)=1

    mask = self.blur_ig_instance.GetMask(self.x_input_val, 
                                         feed_dict={}, 
                                         max_sigma=self.max_sigma, 
                                         steps=x_steps,
                                         batch_size=batch_size)

    # Verify the result.
    np.testing.assert_almost_equal(mask, self.expected_val, decimal=2)
    self.assertEqual(self.sess_spy.run.call_count, expected_calls)

  def testBlurIGGetMaskArgs(self):
    x_steps = 5
    feed_dict = {'foo':'bar'}
    self.sess_spy.run.return_value = [self.x_input_val.reshape((1,5,5,1))]

    self.blur_ig_instance.GetMask(self.x_input_val, 
                                  feed_dict=feed_dict, 
                                  max_sigma=self.max_sigma, 
                                  steps=x_steps)
    actual_feed_dict = self.sess_spy.run.call_args[1]['feed_dict']

    self.assertEqual(actual_feed_dict['foo'], feed_dict['foo'])

if __name__ == '__main__':
  unittest.main()

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

from ..core import blur_ig
import numpy as np
import tensorflow.compat.v1 as tf

OUTPUT_GRADIENTS = blur_ig.OUTPUT_GRADIENTS


class BlurIgTest(unittest.TestCase):
  """To run: "python -m saliency.blur_ig_test" top-level saliency directory."""

  def setUp(self):
    super().setUp()
    self.max_sigma = 10
    with tf.Graph().as_default() as graph:
      self.x = tf.placeholder(shape=[None, 5, 5, 1], dtype=tf.float32)
      # Define function to just look at center pixel.
      y = self.x[:, 2, 2, 0] * 1.0 + tf.sin(self.x[:, 1, 1, 0])
      self.gradients_node = tf.gradients(y, self.x)[0]
      self.sess = tf.Session(graph=graph)
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
      x_baseline_val = blur_ig.gaussian_blur(self.x_input_val,
                                             sigma=self.max_sigma)
      y_baseline_val = self.sess.run(y, feed_dict={self.x: [x_baseline_val]})

      # Test if completeness axiom is satisfied.
      # The expected BlurIG value is equal to the difference between
      # the `y` value at the input and the `y` value at the baseline.
      self.expected_val = y_input_val[0] - y_baseline_val[0]
      self.blur_ig_instance = blur_ig.BlurIG()

  def create_call_model_function(self, session, grad_node, x):
    def call_model(x_value_batch, call_model_args={}, expected_keys=None):
      call_model.num_calls += 1
      call_model_args[x] = x_value_batch
      data = session.run(grad_node, feed_dict=call_model_args)
      return {OUTPUT_GRADIENTS: data}
    call_model.num_calls = 0

    return call_model

  def create_bad_call_model_function(self, session, grad_node, x):
    def call_model(x_value_batch, call_model_args={}, expected_keys=None):
      call_model.num_calls += 1
      call_model_args[x] = x_value_batch
      data = session.run(grad_node, feed_dict=call_model_args)
      return {OUTPUT_GRADIENTS: data[0]}
    call_model.num_calls = 0

    return call_model

  def testBlurIGGetMask(self):
    x_steps = 2000
    # Create a call_model_function using sess and tensors.
    call_model_function = self.create_call_model_function(
        self.sess, self.gradients_node, self.x)

    # Calculate the Blur IG attribution of the input.
    mask = self.blur_ig_instance.GetMask(
        x_value=self.x_input_val, call_model_function=call_model_function,
        call_model_args={}, max_sigma=self.max_sigma, steps=x_steps)

    # Verify the result for completeness..
    # Expected (for max_sigma=10): 1.4746742
    # mask.sum (for max_sigma=10): 1.4742470...
    self.assertAlmostEqual(self.expected_val, mask.sum(), places=3)
    self.assertEqual(x_steps, call_model_function.num_calls)

  def testBlurIGGetMaskBatched(self):
    x_steps = 2001
    batch_size = 100
    expected_calls = 21  #  batch size is 100, ceil(2001/100)=21
    # Create a call_model_function using sess and tensors.
    call_model_function = self.create_call_model_function(
        self.sess, self.gradients_node, self.x)

    # Calculate the Blur IG attribution of the input.
    mask = self.blur_ig_instance.GetMask(
        x_value=self.x_input_val,
        call_model_function=call_model_function,
        call_model_args={},
        max_sigma=self.max_sigma,
        steps=x_steps,
        batch_size=batch_size)

    # Verify the result for completeness..
    # Expected (for max_sigma=10): 1.4746742
    # mask.sum (for max_sigma=10): 1.4742470...
    self.assertAlmostEqual(self.expected_val, mask.sum(), places=3)
    self.assertEqual(expected_calls, call_model_function.num_calls)

  def testBlurIGGetMaskSingleBatch(self):
    x_steps = 499
    batch_size = 500
    expected_calls = 1  # batch size is 500, ceil(499/500)=1
    # Create a call_model_function using sess and tensors.
    call_model_function = self.create_call_model_function(
        self.sess, self.gradients_node, self.x)

    # Calculate the Blur IG attribution of the input.
    mask = self.blur_ig_instance.GetMask(
        x_value=self.x_input_val,
        call_model_function=call_model_function,
        call_model_args={},
        max_sigma=self.max_sigma,
        steps=x_steps,
        batch_size=batch_size)

    # Verify the result for completeness..
    # Expected (for max_sigma=10): 1.4746742
    # mask.sum (for max_sigma=10): 1.4742470...
    self.assertAlmostEqual(self.expected_val, mask.sum(), places=3)
    self.assertEqual(expected_calls, call_model_function.num_calls)

  def testBlurIGGetMaskError(self):
    x_steps = 2001
    # Create a call_model_function using sess and tensors.
    call_model_function = self.create_bad_call_model_function(
        self.sess, self.gradients_node, self.x)

    with self.assertRaisesRegex(ValueError, blur_ig.SHAPE_ERROR_MESSAGE):
      # Calculate the Blur IG attribution of the input.
      self.blur_ig_instance.GetMask(
          x_value=self.x_input_val,
          call_model_function=call_model_function,
          call_model_args={},
          max_sigma=self.max_sigma,
          steps=x_steps,
          batch_size=100)

if __name__ == '__main__':
  unittest.main()

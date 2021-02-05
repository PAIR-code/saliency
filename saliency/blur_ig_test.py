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

"""Tests completeness axiom for blur_ig.
"""
from . import blur_ig
import numpy as np
from tensorflow import test
import tensorflow.compat.v1 as tf

OUTPUT_GRADIENTS = blur_ig.OUTPUT_GRADIENTS


class BlurIgTest(test.TestCase):
  """To run: "python -m saliency.blur_ig_test" top-level saliency directory."""

  def create_call_model_function(self, session, grad_node, x):
    def call_model(x_value_batch, call_model_args={}, expected_keys=None):
      call_model.num_calls += 1
      call_model_args[x] = x_value_batch
      data = session.run(grad_node, feed_dict=call_model_args)
      return {OUTPUT_GRADIENTS: data}
    call_model.num_calls = 0

    return call_model
  
  def testBlurIGGetMask(self):
    x_steps = 2000

    max_sigma = 10
    with tf.Graph().as_default() as graph:
      x = tf.placeholder(shape=[None, 5, 5, 1], dtype=tf.float32)
      # Define function to just look at center pixel.
      y = x[:, 2, 2, 0] * 1.0 + tf.sin(x[:, 1, 1, 0])
      gradients_node = tf.gradients(y, x)[0]

      with tf.Session() as sess:
        # All black except 1 white pixel at the center.
        x_input_val = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.5, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                               ], dtype=np.float)
        x_input_val = x_input_val.reshape((5, 5, 1))
        # Calculate the value of `y` at the input.
        y_input_val = sess.run(y, feed_dict={x: [x_input_val]})

        # Baseline is the fully blurred version of the input.
        x_baseline_val = blur_ig.gaussian_blur(x_input_val, sigma=max_sigma)
        y_baseline_val = sess.run(y, feed_dict={x: [x_baseline_val]})

        # Test if completeness axiom is satisfied.
        # The expected BlurIG value is equal to the difference between
        # the `y` value at the input and the `y` value at the baseline.
        expected_val = y_input_val[0] - y_baseline_val[0]

        # Create a call_model_function using sess and tensors.
        call_model_function = self.create_call_model_function(
            sess, gradients_node, x)

        # Calculate the Blur IG attribution of the input.
        blur_ig_instance = blur_ig.BlurIG()
        mask = blur_ig_instance.GetMask(
            x_value=x_input_val, call_model_function=call_model_function,
            call_model_args={}, max_sigma=max_sigma, steps=x_steps)
        # Verify the result for completeness..
        # Expected (for max_sigma=10): 1.4746742
        # mask.sum (for max_sigma=10): 1.4742470...
        self.assertAlmostEqual(expected_val, mask.sum(), places=3)
        self.assertEqual(x_steps, call_model_function.num_calls)
  
  def testBlurIGGetMaskBatched(self):
    x_steps = 2001
    expected_calls = 21 # batch size is 100, ceil(2001/100)=21

    max_sigma = 10
    with tf.Graph().as_default() as graph:
      x = tf.placeholder(shape=[None, 5, 5, 1], dtype=tf.float32)
      # Define function to just look at center pixel.
      y = x[:, 2, 2, 0] * 1.0 + tf.sin(x[:, 1, 1, 0])
      gradients_node = tf.gradients(y, x)[0]

      with tf.Session() as sess:
        # All black except 1 white pixel at the center.
        x_input_val = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.5, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                               ], dtype=np.float)
        x_input_val = x_input_val.reshape((5, 5, 1))
        # Calculate the value of `y` at the input.
        y_input_val = sess.run(y, feed_dict={x: [x_input_val]})

        # Baseline is the fully blurred version of the input.
        x_baseline_val = blur_ig.gaussian_blur(x_input_val, sigma=max_sigma)
        y_baseline_val = sess.run(y, feed_dict={x: [x_baseline_val]})

        # Test if completeness axiom is satisfied.
        # The expected BlurIG value is equal to the difference between
        # the `y` value at the input and the `y` value at the baseline.
        expected_val = y_input_val[0] - y_baseline_val[0]

        # Create a call_model_function using sess and tensors.
        call_model_function = self.create_call_model_function(
            sess, gradients_node, x)

        # Calculate the Blur IG attribution of the input.
        blur_ig_instance = blur_ig.BlurIG()
        mask = blur_ig_instance.GetMask(
            x_value=x_input_val, call_model_function=call_model_function,
            call_model_args={}, max_sigma=max_sigma, steps=x_steps, 
            batch_size=100)
        # Verify the result for completeness..
        # Expected (for max_sigma=10): 1.4746742
        # mask.sum (for max_sigma=10): 1.4742470...
        self.assertAlmostEqual(expected_val, mask.sum(), places=3)
        self.assertEqual(expected_calls, call_model_function.num_calls)


if __name__ == '__main__':
  test.main()

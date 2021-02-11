# Copyright 2019 Google Inc. All Rights Reserved.
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
import unittest

from ..core import integrated_gradients
import numpy as np
import tensorflow.compat.v1 as tf

OUTPUT_GRADIENTS = integrated_gradients.OUTPUT_GRADIENTS


class IntegratedGradientsTest(unittest.TestCase):
  """To run: "python -m saliency.integrated_gradients_test" from the top-level directory."""

  def setUp(self):
    super().setUp()
    with tf.Graph().as_default() as graph:
      self.x = tf.placeholder(shape=[None, 3], dtype=tf.float32)
      y = 5 * self.x[:, 0] + self.x[:, 1] * self.x[:, 1] + tf.sin(self.x[:, 2])
      self.gradients_node = tf.gradients(y, self.x)[0]
      self.sess = tf.Session(graph=graph)

      # Calculate the value of `y` at the baseline.
      self.x_baseline_val = np.array([[0.5, 0.8, 1.0]], dtype=np.float)
      y_baseline_val = self.sess.run(y, feed_dict={self.x: self.x_baseline_val})

      # Calculate the value of `y` at the input.
      self.x_input_val = np.array([[1.0, 2.0, 3.0]], dtype=np.float)
      y_input_val = self.sess.run(y, feed_dict={self.x: self.x_input_val})

      # Due to mathematical properties of the integrated gradients,
      # the expected IG value is equal to the difference between
      # the `y` value at the input and the `y` value at the baseline.
      self.expected_val = y_input_val[0] - y_baseline_val[0]

      # Calculate the integrated gradients attribution of the input.
      self.ig_instance = integrated_gradients.IntegratedGradients()

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

  def testIntegratedGradientsGetMask(self):
    x_steps = 1000
    call_model_function = self.create_call_model_function(
        self.sess, self.gradients_node, self.x)

    mask = self.ig_instance.GetMask(x_value=self.x_input_val[0],
                                    call_model_function=call_model_function,
                                    call_model_args={},
                                    x_baseline=self.x_baseline_val[0],
                                    x_steps=x_steps)

    # Verify the result.
    self.assertAlmostEqual(self.expected_val, mask.sum(), places=3)
    self.assertEqual(x_steps, call_model_function.num_calls)

  def testIntegratedGradientsGetMaskBatched(self):
    x_steps = 1001
    batch_size = 500
    expected_calls = 3  # batch size is 500, ceil(1001/500)=3

    call_model_function = self.create_call_model_function(
        self.sess, self.gradients_node, self.x)

    mask = self.ig_instance.GetMask(x_value=self.x_input_val[0],
                                    call_model_function=call_model_function,
                                    call_model_args={},
                                    x_baseline=self.x_baseline_val[0],
                                    x_steps=x_steps,
                                    batch_size=batch_size)

    # Verify the result.
    self.assertAlmostEqual(self.expected_val, mask.sum(), places=3)
    self.assertEqual(expected_calls, call_model_function.num_calls)

  def testIntegratedGradientsGetMaskSingleBatch(self):
    x_steps = 999
    batch_size = 1000
    expected_calls = 1  # batch size is 1000, ceil(999/1000)=1

    call_model_function = self.create_call_model_function(
        self.sess, self.gradients_node, self.x)

    mask = self.ig_instance.GetMask(x_value=self.x_input_val[0],
                                    call_model_function=call_model_function,
                                    call_model_args={},
                                    x_baseline=self.x_baseline_val[0],
                                    x_steps=x_steps,
                                    batch_size=batch_size)

    # Verify the result.
    self.assertAlmostEqual(self.expected_val, mask.sum(), places=3)
    self.assertEqual(expected_calls, call_model_function.num_calls)

  def testIntegratedGradientsGetMaskError(self):
    x_steps = 2001
    # Create a call_model_function using sess and tensors.
    call_model_function = self.create_bad_call_model_function(
        self.sess, self.gradients_node, self.x)

    with self.assertRaisesRegex(
        ValueError, integrated_gradients.SHAPE_ERROR_MESSAGE):

      self.ig_instance.GetMask(x_value=self.x_input_val[0],
                               call_model_function=call_model_function,
                               call_model_args={},
                               x_baseline=self.x_baseline_val[0],
                               x_steps=x_steps,
                               batch_size=500)


if __name__ == '__main__':
  unittest.main()

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

from . import integrated_gradients
import numpy as np
import tensorflow.compat.v1 as tf


class IntegratedGradientsTest(unittest.TestCase):
  """To run: "python -m saliency.integrated_gradients_test" from the top-level directory."""

  def setUp(self):
    super().setUp()
    with tf.Graph().as_default() as graph:
      x = tf.placeholder(shape=[None, 3], dtype=tf.float32)
      contrib = [5 * x[:, 0], x[:, 1] * x[:, 1], tf.sin(x[:, 2])]
      y = contrib[0] + contrib[1] + contrib[2]
      sess = tf.Session(graph=graph)
      self.sess_spy = unittest.mock.MagicMock(wraps=sess)

      # Calculate the value of `y` at the baseline.
      self.x_baseline_val = np.array([[0.5, 0.8, 1.0]], dtype=np.float)

      # Calculate the value of `y` at the input.
      self.x_input_val = np.array([[1.0, 2.0, 3.0]], dtype=np.float)


      # Due to mathematical properties of the integrated gradients,
      # the expected IG value is equal to the difference between
      # the `y` value at the input and the `y` value at the baseline.
      contrib_baseline_val = sess.run(contrib, feed_dict={x: self.x_baseline_val})
      contrib_input_val = sess.run(contrib, feed_dict={x: self.x_input_val})

      self.expected_val = np.array(contrib_input_val) - np.array(contrib_baseline_val)
      self.expected_val = self.expected_val.flatten()

      # Calculate the integrated gradients attribution of the input.
      self.ig_instance = integrated_gradients.IntegratedGradients(graph,
                                                                  self.sess_spy,
                                                                  y,
                                                                  x)

  def testIntegratedGradientsGetMask(self):
    x_steps = 1000

    mask = self.ig_instance.GetMask(x_value=self.x_input_val[0],
                                    feed_dict={},
                                    x_baseline=self.x_baseline_val[0],
                                    x_steps=x_steps)

    # Verify the result.
    np.testing.assert_almost_equal(mask, self.expected_val, decimal=2)
    self.assertEqual(self.sess_spy.run.call_count, x_steps)

  def testIntegratedGradientsGetMaskBatched(self):
    x_steps = 1001
    batch_size = 500
    expected_calls = 3  # batch size is 500, ceil(1001/500)=3

    mask = self.ig_instance.GetMask(x_value=self.x_input_val[0],
                                    feed_dict={},
                                    x_baseline=self.x_baseline_val[0],
                                    x_steps=x_steps,
                                    batch_size=batch_size)

    # Verify the result.
    np.testing.assert_almost_equal(mask, self.expected_val, decimal=2)
    self.assertEqual(self.sess_spy.run.call_count, expected_calls)

  def testIntegratedGradientsGetMaskSingleBatch(self):
    x_steps = 999
    batch_size = 1000
    expected_calls = 1  # batch size is 1000, ceil(999/1000)=1

    mask = self.ig_instance.GetMask(x_value=self.x_input_val[0],
                                    feed_dict={},
                                    x_baseline=self.x_baseline_val[0],
                                    x_steps=x_steps,
                                    batch_size=batch_size)

    # Verify the result.
    np.testing.assert_almost_equal(mask, self.expected_val, decimal=2)
    self.assertEqual(self.sess_spy.run.call_count, expected_calls)

  def testIntegratedGradientsGetMaskArgs(self):
    x_steps = 5
    feed_dict = {'foo':'bar'}
    self.sess_spy.run.return_value = [self.x_input_val]

    self.ig_instance.GetMask(x_value=self.x_input_val[0],
        feed_dict=feed_dict,
        x_baseline=self.x_baseline_val[0],
        x_steps=x_steps)
    actual_feed_dict = self.sess_spy.run.call_args[1]['feed_dict']

    self.assertEqual(actual_feed_dict['foo'], feed_dict['foo'])


if __name__ == '__main__':
  unittest.main()
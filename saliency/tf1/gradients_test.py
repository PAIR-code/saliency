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

"""Tests accuracy and correct TF1 usage for gradients."""
import unittest
import unittest.mock as mock

from . import gradients
import numpy as np
import tensorflow.compat.v1 as tf


class GradientsTest(unittest.TestCase):
  """To run: "python -m saliency.tf1.igradients_test" from the top-level directory."""

  def setUp(self):
    super().setUp()
    with tf.Graph().as_default() as graph:
      x = tf.placeholder(shape=[None, 3], dtype=tf.float32)
      contrib = [5 * x[:, 0], x[:, 1] * x[:, 1], tf.sin(x[:, 2])]
      y = contrib[0] + contrib[1] + contrib[2]
      sess = tf.Session(graph=graph)
      self.sess_spy = mock.MagicMock(wraps=sess)

      self.x_input_val = np.array([1.0, 2.0, 3.0])

      self.grad_instance = gradients.GradientSaliency(graph,
                                                      self.sess_spy,
                                                      y,
                                                      x)

  def testGradientsGetMask(self):
    """Tests that gradients are fetched and returned correctly."""
    # f(x,y,z) = 5x + y^2, sin(z)
    # d(f(x,y,z)) = [5, 2y, cos(z)]
    expected_val = np.array([5, 2*2, np.cos(3)])

    mask = self.grad_instance.GetMask(x_value=self.x_input_val,
                                     feed_dict=None)

    # Verify the result.
    np.testing.assert_almost_equal(mask, expected_val, decimal=2)
    self.assertEqual(self.sess_spy.run.call_count, 1)

  def testGradientsGetMaskArgs(self):
    """Tests that sess.run receives all inputs."""
    feed_dict = {'foo':'bar'}
    self.sess_spy.run.return_value = np.array([[self.x_input_val]])

    self.grad_instance.GetMask(x_value=self.x_input_val,
        feed_dict=feed_dict)
    actual_feed_dict = self.sess_spy.run.call_args[1]['feed_dict']

    self.assertEqual(actual_feed_dict['foo'], feed_dict['foo'])


if __name__ == '__main__':
  unittest.main()

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

"""Tests creating call_model_function for TF1 models."""
import unittest

from . import utils
import numpy as np
import tensorflow.compat.v1 as tf

IMAGE_SIZE = 299


class UtilsTF1Test(unittest.TestCase):
  """To run: "python -m saliency.tf1.utils_test" from the top-level directory."""

  def setUp(self):
    super().setUp()
    self.graph = tf.Graph()
    self.sess = tf.Session(graph=self.graph)

  def testOutputGradientsSuccess(self):
    """Tests that output layer gradients are returned correctly."""
    with self.graph.as_default():
      x = tf.placeholder(shape=[None, 3], dtype=np.float32)
      y = (5 * x[:, 0] - 3 * x[:, 1]**2)[0]
    x_value = np.array([[0.5, 0.8, 1.0]], dtype=np.float32)
    # because x[1] is squared, gradient should be -3*2x = -3*2*0.8
    expected = np.array([[5, -3*2*0.8, 0]], dtype=np.float32)

    call_model_function = utils.create_tf1_call_model_function(
        self.graph, self.sess, y, x)
    data = call_model_function(x_value, call_model_args=None,
                               expected_keys=[utils.INPUT_OUTPUT_GRADIENTS])
    actual = data[utils.INPUT_OUTPUT_GRADIENTS]

    self.assertIsNone(np.testing.assert_almost_equal(expected, actual))

  def testOutputValuesSuccess(self):
    """Tests that output layer values are returned correctly."""
    with self.graph.as_default():
      x = tf.placeholder(shape=[None, 3], dtype=np.float32)
      y = (5 * x[:, 0] - 3 * x[:, 1]**2)[0]
    x_value = np.array([[0.5, 0.8, 1.0]], dtype=np.float32)
    # because x[1] is squared, gradient should be -3*2x = -3*2*0.8
    (expected) = self.sess.run(y, feed_dict={x: x_value})

    call_model_function = utils.create_tf1_call_model_function(
        self.graph, self.sess, y, x)
    data = call_model_function(
        x_value, call_model_args=None, expected_keys=[utils.OUTPUT_LAYER_VALUES])
    actual = data[utils.OUTPUT_LAYER_VALUES]

    self.assertIsNone(np.testing.assert_almost_equal(expected, actual))

  def testTwoGradientsSuccess(self):
    """Tests that output layer and conv layer gradients are returned correctly."""
    with self.graph.as_default():
      x = tf.placeholder(shape=[None, 3], dtype=np.float32)
      conv_layer = (-2 * x[:, 0] + 5 * x[:, 1] * x[:, 1])
      y = (conv_layer[:] * 2)[0]
    x_value = np.array([[2, 0.5, 19]], dtype=np.float32)
    # because x[1] is squared, gradient should be 5*2x = 5*2*0.5
    expected_conv_gradient = np.array([2], dtype=np.float32)
    expected_output_gradient = np.array([[-2., 5., 0.]]) * 2

    call_model_function = utils.create_tf1_call_model_function(
        self.graph, self.sess, y=y, x=x, conv_layer=conv_layer)
    data = call_model_function(
        x_value,
        call_model_args=None,
        expected_keys=[
            utils.CONVOLUTION_OUTPUT_GRADIENTS, utils.INPUT_OUTPUT_GRADIENTS
        ])
    actual_conv_gradient = data[utils.CONVOLUTION_OUTPUT_GRADIENTS]
    actual_output_gradient = data[utils.INPUT_OUTPUT_GRADIENTS]

    self.assertIsNone(np.testing.assert_almost_equal(
        expected_conv_gradient, actual_conv_gradient))
    self.assertIsNone(np.testing.assert_almost_equal(
        expected_output_gradient, actual_output_gradient))

  def testThreeKeysSuccess(self):
    """Tests that three expected keys are returned correctly."""
    with self.graph.as_default():
      x = tf.placeholder(shape=[None, 3], dtype=np.float32)
      conv_layer = (-2 * x[:, 0] + 5 * x[:, 1] * x[:, 1])
      y = (conv_layer[:] * 2)[0]
    x_value = np.array([[2, 0.5, 19]], dtype=np.float32)
    # because x[1] is squared, gradient should be 5*2x = 5*2*0.2
    expected_conv_layer = [-2*2 + 5*0.5*0.5]
    # because x[1] is squared, gradient should be 5*2x = 5*2*0.5
    expected_conv_gradient = np.array([2], dtype=np.float32)
    expected_output_gradient = np.array([[-2., 5., 0.]]) * 2

    call_model_function = utils.create_tf1_call_model_function(
        self.graph, self.sess, y=y, x=x, conv_layer=conv_layer)
    data = call_model_function(
        x_value,
        call_model_args=None,
        expected_keys=[
            utils.CONVOLUTION_LAYER_VALUES,
            utils.INPUT_OUTPUT_GRADIENTS,
            utils.CONVOLUTION_OUTPUT_GRADIENTS
        ])
    actual_conv_gradient = data[utils.CONVOLUTION_OUTPUT_GRADIENTS]
    actual_output_gradient = data[utils.INPUT_OUTPUT_GRADIENTS]
    actual_conv_layer = data[utils.CONVOLUTION_LAYER_VALUES]

    self.assertIsNone(
        np.testing.assert_almost_equal(expected_conv_gradient,
                                       actual_conv_gradient))
    self.assertIsNone(
        np.testing.assert_almost_equal(expected_conv_layer, actual_conv_layer))
    self.assertIsNone(
        np.testing.assert_almost_equal(expected_output_gradient,
                                       actual_output_gradient))

  def testConvolutionGradientsMissingConvLayer(self):
    """Tests that call_model_function can't get conv values without conv_layer."""
    with self.graph.as_default():
      x = tf.placeholder(shape=[None, 3], dtype=np.float32)
      conv_layer = (-2 * x[:, 0] + 5 * x[:, 1] * x[:, 1])
      y = (conv_layer[:] * 2)[0]
    x_value = np.array([[0.5, 0.8, 1.0]], dtype=float)
    expected = (
        'Cannot return key CONVOLUTION_OUTPUT_GRADIENTS because no conv_layer '
        'was specified'
    )

    with self.assertRaisesRegex(RuntimeError, expected):
      call_model_function = utils.create_tf1_call_model_function(
          self.graph, self.sess, y=y, x=x)
      call_model_function(
          x_value,
          call_model_args=None,
          expected_keys=[utils.CONVOLUTION_OUTPUT_GRADIENTS])

  def testInvalidKey(self):
    """Tests that call_model_function can't get conv values without conv_layer."""
    with self.graph.as_default():
      x = tf.placeholder(shape=[None, 3], dtype=np.float32)
      conv_layer = (-2 * x[:, 0] + 5 * x[:, 1] * x[:, 1])
      y = (conv_layer[:] * 2)[0]
    x_value = np.array([[0.5, 0.8, 1.0]], dtype=float)
    expected = 'Invalid expected key FOO_BAR'

    with self.assertRaisesRegex(ValueError, expected):
      call_model_function = utils.create_tf1_call_model_function(
          self.graph, self.sess, y=y, x=x, conv_layer=conv_layer)
      call_model_function(
          x_value,
          call_model_args=None,
          expected_keys=['FOO_BAR'])


if __name__ == '__main__':
  unittest.main()

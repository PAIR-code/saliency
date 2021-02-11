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

"""Tests creating call_model_function for TF1 models."""
import unittest

from . import utils
import numpy as np
import tensorflow.compat.v1 as tf

IMAGE_SIZE = 299


class UtilsTF1Test(unittest.TestCase):
  """To run: "python -m saliency.xrai_test" from the top-level directory."""

  def setUp(self):
    super().setUp()
    self.graph = tf.Graph()
    self.sess = tf.Session(graph=self.graph)

  def testOutputGradientsSuccess(self):
    with self.graph.as_default():
      x = tf.placeholder(shape=[None, 3], dtype=np.float32)
      y = (5 * x[:, 0] - 3 * x[:, 1]**2)[0]
    x_value = np.array([[0.5, 0.8, 1.0]], dtype=np.float32)
    # because x[1] is squared, gradient should be -3*2x = -3*2*0.8
    expected = np.array([[5, -3*2*0.8, 0]], dtype=np.float32)

    call_model_function = utils.create_tf1_call_model_function(
        self.graph, self.sess, y, x)
    data = call_model_function(
        x_value, call_model_args={}, expected_keys=[utils.OUTPUT_LAYER_GRADIENTS])
    actual = data[utils.OUTPUT_LAYER_GRADIENTS]

    self.assertIsNone(np.testing.assert_almost_equal(expected, actual))

  def testTwoGradientsSuccess(self):
    with self.graph.as_default():
      x = tf.placeholder(shape=[None, 3], dtype=np.float32)
      conv_layer = (-2 * x[:, 0] + 5 * x[:, 1] * x[:, 1]),
      y = (conv_layer[0] * 2)[0]
      # y = (conv_layer[0] + conv_layer[1] + conv_layer[2])[0]
    x_value = np.array([[2, 0.5, 19]], dtype=np.float32)
    # because x[1] is squared, gradient should be 5*2x = 5*2*0.2
    expected_conv_gradient = np.array([[-2, 5*2*0.5, 0]], dtype=np.float32)
    expected_output_gradient = expected_conv_gradient * 2

    call_model_function = utils.create_tf1_call_model_function(
        self.graph, self.sess, y=y, x=x, conv_layer=conv_layer)
    data = call_model_function(
        x_value,
        call_model_args={},
        expected_keys=[
            utils.CONVOLUTION_LAYER_GRADIENTS, utils.OUTPUT_LAYER_GRADIENTS
        ])
    actual_conv_gradient = data[utils.CONVOLUTION_LAYER_GRADIENTS]
    actual_output_gradient = data[utils.OUTPUT_LAYER_GRADIENTS]

    self.assertIsNone(np.testing.assert_almost_equal(
        expected_conv_gradient, actual_conv_gradient))
    self.assertIsNone(np.testing.assert_almost_equal(
        expected_output_gradient, actual_output_gradient))

  def testThreeKeysSuccess(self):
    with self.graph.as_default():
      x = tf.placeholder(shape=[None, 3], dtype=np.float32)
      conv_layer = (-2 * x[:, 0] + 5 * x[:, 1] * x[:, 1])
      y = (conv_layer[:] * 2)[0]
      # y = (conv_layer[0] + conv_layer[1] + conv_layer[2])[0]
    x_value = np.array([[2, 0.5, 19]], dtype=np.float32)
    # because x[1] is squared, gradient should be 5*2x = 5*2*0.2
    expected_conv_gradient = np.array([[-2, 5*2*0.5, 0]], dtype=np.float32)
    expected_conv_layer = [-2*2 + 5*0.5*0.5]
    expected_output_gradient = expected_conv_gradient * 2

    call_model_function = utils.create_tf1_call_model_function(
        self.graph, self.sess, y=y, x=x, conv_layer=conv_layer)
    data = call_model_function(x_value,
                               call_model_args={},
                               expected_keys=[utils.CONVOLUTION_LAYER_VALUES,
                                              utils.OUTPUT_LAYER_GRADIENTS,
                                              utils.CONVOLUTION_LAYER_GRADIENTS])
    actual_conv_gradient = data[utils.CONVOLUTION_LAYER_GRADIENTS]
    actual_output_gradient = data[utils.OUTPUT_LAYER_GRADIENTS]
    actual_conv_layer = data[utils.CONVOLUTION_LAYER_VALUES]

    self.assertIsNone(
        np.testing.assert_almost_equal(expected_conv_gradient,
                                       actual_conv_gradient))
    self.assertIsNone(
        np.testing.assert_almost_equal(expected_conv_layer, actual_conv_layer))
    self.assertIsNone(
        np.testing.assert_almost_equal(expected_output_gradient,
                                       actual_output_gradient))

  def testOutputGradientsMissingY(self):
    with self.graph.as_default():
      x = tf.placeholder(shape=[None, 3], dtype=np.float32)
    x_value = np.array([[0.5, 0.8, 1.0]], dtype=np.float)
    expected = 'Cannot return key OUTPUT_LAYER_GRADIENTS because no y was specified'

    with self.assertRaisesRegex(RuntimeError, expected):
      call_model_function = utils.create_tf1_call_model_function(
          self.graph, self.sess, x=x)
      call_model_function(
          x_value,
          call_model_args={},
          expected_keys=[utils.OUTPUT_LAYER_GRADIENTS])

  def testMissingX(self):
    with self.graph.as_default():
      x = tf.placeholder(shape=[None, 3], dtype=np.float32)
      y = (5 * x[:, 0] - 3 * x[:, 1])[0]
    expected = 'Expected input tensor for x but is equal to None'

    with self.assertRaisesRegex(ValueError, expected):
      utils.create_tf1_call_model_function(self.graph, self.sess, y=y)

  def testConvolutionGradientsMissingConvLayer(self):
    with self.graph.as_default():
      x = tf.placeholder(shape=[None, 3], dtype=np.float32)
    x_value = np.array([[0.5, 0.8, 1.0]], dtype=np.float)
    expected = ('Cannot return key CONVOLUTION_LAYER_GRADIENTS because no conv_layer '
                'was specified')

    with self.assertRaisesRegex(RuntimeError, expected):
      call_model_function = utils.create_tf1_call_model_function(
          self.graph, self.sess, x=x)
      call_model_function(
          x_value,
          call_model_args={},
          expected_keys=[utils.CONVOLUTION_LAYER_GRADIENTS])

  def testConvolutionLayerMissingConvLayer(self):
    with self.graph.as_default():
      x = tf.placeholder(shape=[None, 3], dtype=np.float32)
    x_value = np.array([[0.5, 0.8, 1.0]], dtype=np.float)
    expected = ('Cannot return key CONVOLUTION_LAYER_VALUES because no conv_layer was '
                'specified')

    with self.assertRaisesRegex(RuntimeError, expected):
      call_model_function = utils.create_tf1_call_model_function(
          self.graph, self.sess, x=x)
      call_model_function(
          x_value,
          call_model_args={},
          expected_keys=[utils.CONVOLUTION_LAYER_VALUES])


if __name__ == '__main__':
  unittest.main()

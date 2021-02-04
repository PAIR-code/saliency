from unittest import mock
import unittest
import numpy as np
import tensorflow.compat.v1 as tf
from . import utils_tf1

IMAGE_SIZE = 299


class UtilsTF1Test(unittest.TestCase):
  """To run: "python -m saliency.xrai_test" from the top-level directory."""

  def setUp(self):
    super().setUp()
    self.graph = tf.Graph()
    self.sess = tf.Session(graph=self.graph)
    
  def tearDown(self):
    super().tearDown()
    # self.mock_ig.stop()
    b = 2

  def testOutputGradientsSuccess(self):
    with self.graph.as_default():
      x = tf.placeholder(shape=[None, 3], dtype=np.float32)
      y = (5 * x[:, 0] - 3 * x[:, 1]**2)[0]
    x_value = np.array([[0.5, 0.8, 1.0]], dtype=np.float32)
    # because x[1] is squared, gradient should be -3*2x = -3*2*0.8
    expected = np.array([[5, -3*2*0.8, 0]], dtype=np.float32)

    call_model_function = utils_tf1.create_tf1_call_model_function(self.graph,
      self.sess, y, x)
    data = call_model_function(x_value, call_model_args={},
      expected_keys=[utils_tf1.OUTPUT_GRADIENTS])
    actual = data[utils_tf1.OUTPUT_GRADIENTS]

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

    call_model_function = utils_tf1.create_tf1_call_model_function(self.graph,
      self.sess, y=y, x=x, conv_layer=conv_layer)
    data = call_model_function(x_value, call_model_args={},
      expected_keys=[utils_tf1.CONVOLUTION_GRADIENTS, utils_tf1.OUTPUT_GRADIENTS])
    actual_conv_gradient = data[utils_tf1.CONVOLUTION_GRADIENTS]
    actual_output_gradient = data[utils_tf1.OUTPUT_GRADIENTS]

    self.assertIsNone(np.testing.assert_almost_equal(expected_conv_gradient,
      actual_conv_gradient))
    self.assertIsNone(np.testing.assert_almost_equal(expected_output_gradient,
      actual_output_gradient))

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

    call_model_function = utils_tf1.create_tf1_call_model_function(self.graph,
      self.sess, y=y, x=x, conv_layer=conv_layer)
    data = call_model_function(x_value, call_model_args={},
      expected_keys=[utils_tf1.CONVOLUTION_LAYER, 
        utils_tf1.OUTPUT_GRADIENTS,
        utils_tf1.CONVOLUTION_GRADIENTS])
    actual_conv_gradient = data[utils_tf1.CONVOLUTION_GRADIENTS]
    actual_output_gradient = data[utils_tf1.OUTPUT_GRADIENTS]
    actual_conv_layer = data[utils_tf1.CONVOLUTION_LAYER]

    self.assertIsNone(np.testing.assert_almost_equal(expected_conv_gradient,
      actual_conv_gradient))
    self.assertIsNone(np.testing.assert_almost_equal(expected_conv_layer,
      actual_conv_layer))
    self.assertIsNone(np.testing.assert_almost_equal(expected_output_gradient,
      actual_output_gradient))

  def testOutputGradientsMissingY(self):
    with self.graph.as_default():
      x = tf.placeholder(shape=[None, 3], dtype=np.float32)
    x_value = np.array([[0.5, 0.8, 1.0]], dtype=np.float)
    expected = 'Cannot return key OUTPUT_GRADIENTS because no y was specified'

    with self.assertRaisesRegex(RuntimeError, expected):
      call_model_function = utils_tf1.create_tf1_call_model_function(self.graph,
        self.sess, x=x)
      call_model_function(x_value, call_model_args={},
        expected_keys=[utils_tf1.OUTPUT_GRADIENTS])

  def testMissingX(self):
    with self.graph.as_default():
      x = tf.placeholder(shape=[None, 3], dtype=np.float32)
      y = (5 * x[:, 0] - 3 * x[:, 1])[0]
    expected = 'Expected input tensor for x but is equal to None'

    with self.assertRaisesRegex(ValueError, expected):
      utils_tf1.create_tf1_call_model_function(self.graph,
        self.sess, y=y)

  def testConvolutionGradientsMissingConvLayer(self):
    with self.graph.as_default():
      x = tf.placeholder(shape=[None, 3], dtype=np.float32)
    x_value = np.array([[0.5, 0.8, 1.0]], dtype=np.float)
    expected = 'Cannot return key CONVOLUTION_GRADIENTS because no conv_layer was specified'

    with self.assertRaisesRegex(RuntimeError, expected):
      call_model_function = utils_tf1.create_tf1_call_model_function(self.graph,
        self.sess, x=x)
      call_model_function(x_value, call_model_args={},
        expected_keys=[utils_tf1.CONVOLUTION_GRADIENTS])

  def testConvolutionLayerMissingConvLayer(self):
    with self.graph.as_default():
      x = tf.placeholder(shape=[None, 3], dtype=np.float32)
    x_value = np.array([[0.5, 0.8, 1.0]], dtype=np.float)
    expected = 'Cannot return key CONVOLUTION_LAYER because no conv_layer was specified'

    with self.assertRaisesRegex(RuntimeError, expected):
      call_model_function = utils_tf1.create_tf1_call_model_function(self.graph,
        self.sess, x=x)
      call_model_function(x_value, call_model_args={},
        expected_keys=[utils_tf1.CONVOLUTION_LAYER])


if __name__ == '__main__':
  unittest.main()

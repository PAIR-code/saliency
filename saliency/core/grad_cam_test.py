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

"""Tests accuracy and error handling for grad_cam."""
import unittest
import unittest.mock as mock

from .base import SHAPE_ERROR_MESSAGE
from . import grad_cam
import numpy as np

CONVOLUTION_LAYER_VALUES = grad_cam.CONVOLUTION_LAYER_VALUES
CONVOLUTION_OUTPUT_GRADIENTS = grad_cam.CONVOLUTION_OUTPUT_GRADIENTS
INPUT_HEIGHT_WIDTH = 5  # width and height of input images in pixels


class GradCamTest(unittest.TestCase):
  """To run: "python -m saliency.core.grad_cam_test" from top-level saliency directory."""

  def setUp(self):
    super().setUp()
    self.grad_cam_instance = grad_cam.GradCam()

  def testGradCamGetMask(self):
    """Tests the GradCAM method using a simple network.

    Simple test case where the network contains one convolutional layer that
    acts as a horizontal line detector and the input image is a 5x5 matrix with
    a centered 3x3 grid of 1s and 0s elsewhere.

    The computed GradCAM mask should detect the pixels of highest importance to
    be along the two horizontal lines in the image (exact expected values stored
    in ref_mask).
    """

    def create_call_model_function():

      def call_model(x_value_batch, call_model_args=None, expected_keys=None):
        # simulates conv layer output and grads where the kernel for the conv
        # layer is a horizontal line detector of kernel size 3 and the input is
        # a 3x3 square of ones in the center of the image.
        grad = np.zeros([5, 5])
        grad[(0, -1), (0, -1)] = 2
        grad[(1, -1), 1:-1] = 3
        output = np.zeros([5, 5])
        output[:] = [1, 2, 3, 2, 1]
        output[(0, -1), :] *= -1
        output[2, :] = 0
        grad = grad.reshape(x_value_batch.shape)
        output = output.reshape(x_value_batch.shape)
        return {CONVOLUTION_LAYER_VALUES: output,
                CONVOLUTION_OUTPUT_GRADIENTS: grad}

      return call_model

    call_model_function = create_call_model_function()

    # Generate test input (centered matrix of 1s surrounded by 0s)
    # and generate corresponding GradCAM mask
    img = np.zeros([INPUT_HEIGHT_WIDTH, INPUT_HEIGHT_WIDTH])
    img[1:-1, 1:-1] = 1
    img = img.reshape([INPUT_HEIGHT_WIDTH, INPUT_HEIGHT_WIDTH, 1])
    mask = self.grad_cam_instance.GetMask(
        img,
        call_model_function=call_model_function,
        call_model_args=None,
        should_resize=True,
        three_dims=False)

    # Compare generated mask to expected result
    ref_mask = np.array([[0., 0., 0., 0., 0.],
                         [0.33, 0.67, 1., 0.67, 0.33],
                         [0., 0., 0., 0., 0.],
                         [0.33, 0.67, 1., 0.67, 0.33],
                         [0., 0., 0., 0., 0.]])
    self.assertTrue(
        np.allclose(mask, ref_mask, atol=0.01),
        'Generated mask did not match reference mask.')

  def testGradCamCallModelArgs(self):
    img = np.ones([INPUT_HEIGHT_WIDTH, INPUT_HEIGHT_WIDTH])
    img = img.reshape([INPUT_HEIGHT_WIDTH, INPUT_HEIGHT_WIDTH, 1])
    expected_keys = [
        CONVOLUTION_LAYER_VALUES, CONVOLUTION_OUTPUT_GRADIENTS
    ]
    call_model_args = {'foo': 'bar'}
    mock_call_model = mock.MagicMock(
        return_value={
            CONVOLUTION_LAYER_VALUES: [img],
            CONVOLUTION_OUTPUT_GRADIENTS: [img]
        })

    self.grad_cam_instance.GetMask(
        img,
        call_model_function=mock_call_model,
        call_model_args=call_model_args,
        should_resize=True,
        three_dims=False)
    calls = mock_call_model.mock_calls

    self.assertEqual(len(calls), 1)
    for call in calls:
      kwargs = call[2]
      self.assertEqual(
          kwargs['call_model_args'],
          call_model_args,
          msg='function was called with incorrect call_model_args.')
      self.assertEqual(
          kwargs['expected_keys'],
          expected_keys,
          msg='function was called with incorrect expected_keys.')

  def testGradCamErrorGradientsMismatch(self):
    """Tests the GradCAM method using a simple network.

    Simple test case where the network contains one convolutional layer that
    acts as a horizontal line detector and the input image is a 5x5 matrix with
    a centered 3x3 grid of 1s and 0s elsewhere.

    The call_model_function returns the gradients without the outermost batch
    dimension, so the expectation is that a ValueError will be raised.
    """

    def create_call_model_function():

      def call_model(x_value_batch, call_model_args=None, expected_keys=None):
        grad = np.zeros(x_value_batch.shape)
        output = np.zeros(x_value_batch.shape)
        return {CONVOLUTION_LAYER_VALUES: output,
                CONVOLUTION_OUTPUT_GRADIENTS: grad[0]}

      return call_model

    call_model_function = create_call_model_function()

    # Generate test input (centered matrix of 1s surrounded by 0s)
    # and generate corresponding GradCAM mask
    img = np.zeros([INPUT_HEIGHT_WIDTH, INPUT_HEIGHT_WIDTH])
    img[1:-1, 1:-1] = 1
    img = img.reshape([INPUT_HEIGHT_WIDTH, INPUT_HEIGHT_WIDTH, 1])
    expected_error = SHAPE_ERROR_MESSAGE[CONVOLUTION_OUTPUT_GRADIENTS].format(
        '1', '5')

    with self.assertRaisesRegex(ValueError, expected_error):

      self.grad_cam_instance.GetMask(
          img,
          call_model_function=call_model_function,
          call_model_args=None,
          should_resize=True,
          three_dims=False)

  def testGradCamErrorValuesMismatch(self):
    """Tests the GradCAM method using a simple network.

    Simple test case where the network contains one convolutional layer that
    acts as a horizontal line detector and the input image is a 5x5 matrix with
    a centered 3x3 grid of 1s and 0s elsewhere.

    The call_model_function returns the gradients without the outermost batch
    dimension, so the expectation is that a ValueError will be raised.
    """

    def create_call_model_function():

      def call_model(x_value_batch, call_model_args=None, expected_keys=None):
        grad = np.zeros(x_value_batch.shape)
        output = np.zeros(x_value_batch.shape)
        return {CONVOLUTION_OUTPUT_GRADIENTS: grad,
                CONVOLUTION_LAYER_VALUES: output[0]}

      return call_model

    call_model_function = create_call_model_function()

    # Generate test input (centered matrix of 1s surrounded by 0s)
    # and generate corresponding GradCAM mask
    img = np.zeros([INPUT_HEIGHT_WIDTH, INPUT_HEIGHT_WIDTH])
    img[1:-1, 1:-1] = 1
    img = img.reshape([INPUT_HEIGHT_WIDTH, INPUT_HEIGHT_WIDTH, 1])
    expected_error = SHAPE_ERROR_MESSAGE[CONVOLUTION_LAYER_VALUES].format('1', '5')

    with self.assertRaisesRegex(ValueError, expected_error):
      self.grad_cam_instance.GetMask(
          img,
          call_model_function=call_model_function,
          call_model_args=None,
          should_resize=True,
          three_dims=False)

if __name__ == '__main__':
  unittest.main()

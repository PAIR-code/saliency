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

"""Tests accuracy and error handling for occlusion."""
import unittest
import unittest.mock as mock

from .base import OUTPUT_LAYER_VALUES
from .base import SHAPE_ERROR_MESSAGE
from . import occlusion
import numpy as np

INPUT_HEIGHT_WIDTH = 5  # width and height of input images in pixels


class OcclusionTest(unittest.TestCase):
  """To run: "python -m saliency.core.occlusion_test" from top-level saliency directory."""

  def setUp(self):
    super().setUp()
    self.occlusion_instance = occlusion.Occlusion()

  def testOcclusionGetMask(self):
    """Tests the Occlusion method using a simple network.

    Simple test case where the network multiplies all values by 3, and the input
    image has all zero values except at positions [1,1] and [3,3].

    The computed Occlusion mask should bias the center, but the quadrants with
    nonzero values should be greater than the other two (exact expected values
    stored in ref_mask).
    """

    def create_call_model_function():

      def call_model(x_value_batch, call_model_args=None, expected_keys=None):
        # simulates output where all values are multiplied by 3.
        output = [np.sum(x_value_batch) * 3]
        return {occlusion.OUTPUT_LAYER_VALUES: output}

      return call_model
    call_model_function = create_call_model_function()
    # Generate test input and generate corresponding Occlusion mask
    img = np.zeros([INPUT_HEIGHT_WIDTH, INPUT_HEIGHT_WIDTH])
    img[1, 1] = 1
    img[3, 3] = 1
    img = img.reshape((INPUT_HEIGHT_WIDTH, INPUT_HEIGHT_WIDTH))
    ref_mask = np.array([[3., 6., 6., 3., 0.],
                         [6., 15., 18., 12., 3.],
                         [6., 18., 24., 18., 6.],
                         [3., 12., 18., 15., 6.],
                         [0., 3., 6., 6., 3.]])

    mask = self.occlusion_instance.GetMask(
        img,
        call_model_function=call_model_function,
        call_model_args=None,
        size=3,
        value=0)

    # Compare generated mask to expected result
    self.assertTrue(np.allclose(mask, ref_mask, atol=0.01),
                    'Generated mask did not match reference mask.')

  def testOcclusionCallModelArgs(self):
    """Tests the call_model_function receives the correct inputs."""
    img = np.ones([INPUT_HEIGHT_WIDTH, INPUT_HEIGHT_WIDTH])
    expected_keys = [occlusion.OUTPUT_LAYER_VALUES]
    call_model_args = {'foo': 'bar'}
    mock_call_model = mock.MagicMock(
        return_value={occlusion.OUTPUT_LAYER_VALUES: [3]})

    self.occlusion_instance.GetMask(
        img,
        call_model_function=mock_call_model,
        call_model_args=call_model_args,
        size=3,
        value=0)
    calls = mock_call_model.mock_calls

    self.assertEqual(len(calls), 10)
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

  def testOcclusionValuesMismatch(self):
    """Tests the Occlusion method errors with incorrect model outputs."""

    def create_call_model_function():
      # Bad call model function since the expected output is a single value
      def call_model(x_value_batch, call_model_args=None, expected_keys=None):
        output = np.ones(INPUT_HEIGHT_WIDTH)
        return {occlusion.OUTPUT_LAYER_VALUES: output}

      return call_model
    call_model_function = create_call_model_function()
    img = np.zeros([INPUT_HEIGHT_WIDTH, INPUT_HEIGHT_WIDTH])
    expected_error = SHAPE_ERROR_MESSAGE[OUTPUT_LAYER_VALUES].format('1', '5')

    with self.assertRaisesRegex(ValueError, expected_error):
      self.occlusion_instance.GetMask(
          img,
          call_model_function=call_model_function,
          call_model_args=None,
          size=3,
          value=0)


if __name__ == '__main__':
  unittest.main()

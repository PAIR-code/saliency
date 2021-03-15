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

"""Tests accuracy and error handling for gradients."""
import unittest
import unittest.mock as mock

from .base import INPUT_OUTPUT_GRADIENTS
from .base import SHAPE_ERROR_MESSAGE
from . import gradients
import numpy as np


class GradientSaliencyTest(unittest.TestCase):
  """To run: "python -m saliency.core.gradients_test" from the top-level directory."""

  def gradient_fn(self, arr):
    # f(x,y,z) = 5x + y^2, sin(z)
    # d(f(x,y,z)) = [5, 2y, cos(z)]
    return np.array([5, 2*arr[1], np.cos(arr[2])])

  def create_call_model_function(self):
    def call_model(x_value_batch, call_model_args=None, expected_keys=None):
      call_model.num_calls += 1
      data = np.apply_along_axis(self.gradient_fn, 1, x_value_batch)
      return {INPUT_OUTPUT_GRADIENTS: data}
    call_model.num_calls = 0

    return call_model

  def create_bad_call_model_function(self):
    # Bad call model function since gradient shape doesn't match input.
    def call_model(x_value_batch, call_model_args=None, expected_keys=None):
      call_model.num_calls += 1
      data = np.apply_along_axis(self.gradient_fn, 1, x_value_batch)
      return {INPUT_OUTPUT_GRADIENTS: data[0]}
    call_model.num_calls = 0

    return call_model

  def testGradientsGetMask(self):
    """Tests that GradientSaliency returns the output gradients."""
    call_model_function = self.create_call_model_function()
    grad_instance = gradients.GradientSaliency()
    x_input = [3, 2, 1]
    self.expected_val = self.gradient_fn(x_input)

    mask = grad_instance.GetMask(x_value=x_input,
                                 call_model_function=call_model_function,
                                 call_model_args=None)

    # Verify the result.
    np.testing.assert_almost_equal(mask, self.expected_val, decimal=2)
    self.assertEqual(call_model_function.num_calls, 1)

  def testGradientCallModelArgs(self):
    """Tests that call_model_function receives all inputs."""
    expected_keys = [INPUT_OUTPUT_GRADIENTS]
    call_model_args = {'foo': 'bar'}
    x_input = [3, 2, 1]
    mock_call_model = mock.MagicMock(
        return_value={INPUT_OUTPUT_GRADIENTS: [x_input]})
    grad_instance = gradients.GradientSaliency()

    grad_instance.GetMask(x_value=x_input,
                          call_model_function=mock_call_model,
                          call_model_args=call_model_args)
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

  def testGradientsGetMaskError(self):
    """Tests that GradientSaliency errors when receiving incorrect model output."""
    call_model_function = self.create_bad_call_model_function()
    expected_error = SHAPE_ERROR_MESSAGE[INPUT_OUTPUT_GRADIENTS].format(
        '\\(1, 3\\)', '\\(3,\\)')
    grad_instance = gradients.GradientSaliency()
    x_input = [3, 2, 1]

    with self.assertRaisesRegex(ValueError, expected_error):

      grad_instance.GetMask(x_value=x_input,
                            call_model_function=call_model_function,
                            call_model_args=None)


if __name__ == '__main__':
  unittest.main()

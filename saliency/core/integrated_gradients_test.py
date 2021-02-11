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

OUTPUT_LAYER_GRADIENTS = integrated_gradients.OUTPUT_LAYER_GRADIENTS


class IntegratedGradientsTest(unittest.TestCase):
  """To run: "python -m saliency.integrated_gradients_test" from the top-level directory."""

  def setUp(self):
    super().setUp()
    def y_fn(arr):
      return np.array([5 * arr[0], arr[1]*arr[1], np.sin(arr[2])])

    # Calculate the value of `y` at the baseline. `y` is given as multiple 
    # numbers, with the assumption that in a real network these values would be 
    # summed to a single output number.
    self.x_baseline_val = np.array([0.5, 1.0, 1.0], dtype=np.float)
    y_baseline_val = y_fn(self.x_baseline_val)

    # Calculate the value of `y` at the input.
    self.x_input_val = np.array([1.0, 2.0, 3.0], dtype=np.float)
    y_input_val = y_fn(self.x_input_val)

    # Because each variable is independent, the expected contribution is equal
    # to the difference between the baseline and input contribution for each.
    self.expected_val = y_input_val - y_baseline_val

    # Calculate the integrated gradients attribution of the input.
    self.ig_instance = integrated_gradients.IntegratedGradients()

  def create_call_model_function(self):

    def gradient_fn(arr):
      # f(x,y,z) = 5x + y^2, sin(z)
      # d(f(x,y,z)) = [5, 2y, cos(z)]
      return np.array([5, 2*arr[1], np.cos(arr[2])])

    def call_model(x_value_batch, call_model_args={}, expected_keys=None):
      call_model.num_calls += 1
      data = np.apply_along_axis(gradient_fn, 1, x_value_batch)
      return {OUTPUT_LAYER_GRADIENTS: data}
    call_model.num_calls = 0

    return call_model

  def create_bad_call_model_function(self):

    def gradient_fn(arr):
      return np.array([5, 2*arr[1], np.cos(arr[2])])

    def call_model(x_value_batch, call_model_args={}, expected_keys=None):
      call_model.num_calls += 1
      data = np.apply_along_axis(gradient_fn, 1, x_value_batch)
      return {OUTPUT_LAYER_GRADIENTS: data[0]}
    call_model.num_calls = 0

    return call_model

  def testIntegratedGradientsGetMask(self):
    x_steps = 1000
    call_model_function = self.create_call_model_function()

    mask = self.ig_instance.GetMask(x_value=self.x_input_val,
                                    call_model_function=call_model_function,
                                    call_model_args={},
                                    x_baseline=self.x_baseline_val,
                                    x_steps=x_steps)

    # Verify the result.
    np.testing.assert_almost_equal(mask, self.expected_val, decimal=2)
    self.assertEqual(x_steps, call_model_function.num_calls)

  def testIntegratedGradientsGetMaskBatched(self):
    x_steps = 1001
    batch_size = 500
    expected_calls = 3  # batch size is 500, ceil(1001/500)=3

    call_model_function = self.create_call_model_function()

    mask = self.ig_instance.GetMask(x_value=self.x_input_val,
                                    call_model_function=call_model_function,
                                    call_model_args={},
                                    x_baseline=self.x_baseline_val,
                                    x_steps=x_steps,
                                    batch_size=batch_size)

    # Verify the result.
    np.testing.assert_almost_equal(mask, self.expected_val, decimal=2)
    self.assertEqual(expected_calls, call_model_function.num_calls)

  def testIntegratedGradientsGetMaskSingleBatch(self):
    x_steps = 999
    batch_size = 1000
    expected_calls = 1  # batch size is 1000, ceil(999/1000)=1

    call_model_function = self.create_call_model_function()

    mask = self.ig_instance.GetMask(x_value=self.x_input_val,
                                    call_model_function=call_model_function,
                                    call_model_args={},
                                    x_baseline=self.x_baseline_val,
                                    x_steps=x_steps,
                                    batch_size=batch_size)

    # Verify the result.
    np.testing.assert_almost_equal(mask, self.expected_val, decimal=2)
    self.assertEqual(expected_calls, call_model_function.num_calls)

  def testBlurIGCallModelArgs(self):
    x_steps = 50
    expected_keys = [OUTPUT_LAYER_GRADIENTS]
    call_model_args = {'foo': 'bar'}
    mock_call_model = unittest.mock.MagicMock(
        return_value={OUTPUT_LAYER_GRADIENTS: [self.x_input_val]})

    self.ig_instance.GetMask(
        x_value=self.x_input_val,
        call_model_function=mock_call_model,
        call_model_args=call_model_args,
        x_steps=x_steps)
    calls = mock_call_model.mock_calls

    self.assertEqual(len(calls), x_steps)
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

  def testIntegratedGradientsGetMaskError(self):
    x_steps = 2001
    # Create a call_model_function using sess and tensors.
    call_model_function = self.create_bad_call_model_function()

    with self.assertRaisesRegex(
        ValueError, integrated_gradients.SHAPE_ERROR_MESSAGE[:-30]):

      self.ig_instance.GetMask(x_value=self.x_input_val,
                               call_model_function=call_model_function,
                               call_model_args={},
                               x_baseline=self.x_baseline_val,
                               x_steps=x_steps,
                               batch_size=500)


if __name__ == '__main__':
  unittest.main()
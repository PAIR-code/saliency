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

"""Tests completeness axiom, batching, and error handling for blur_ig.
"""
import unittest

from . import blur_ig
import numpy as np

OUTPUT_LAYER_GRADIENTS = blur_ig.OUTPUT_LAYER_GRADIENTS


class BlurIgTest(unittest.TestCase):
  """To run: "python -m saliency.blur_ig_test" top-level saliency directory."""

  def setUp(self):
    super().setUp()
    self.max_sigma = 10
    # All black except 1 white pixel at the center.
    self.x_input_val = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.7, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ],
                                dtype=np.float)
    self.x_input_val = self.x_input_val.reshape((5, 5, 1))
    # Calculate the value of `y` at the input.
    y_input_val = np.sin(self.x_input_val)

    # Baseline is the fully blurred version of the input.
    self.x_baseline_val = blur_ig.gaussian_blur(self.x_input_val,
                                            sigma=self.max_sigma)
    y_baseline_val = np.sin(self.x_baseline_val)

    # Test if completeness axiom is satisfied.
    # The expected BlurIG value is equal to the difference between
    # the `y` value at the input and the `y` value at the baseline.
    self.expected_val = y_input_val - y_baseline_val
    self.expected_argmax = 12  # middle of the 5x5 matrix
    self.blur_ig_instance = blur_ig.BlurIG()

  def create_call_model_function(self):
    def call_model(x_value_batch, call_model_args={}, expected_keys=None):
      call_model.num_calls += 1
      return {OUTPUT_LAYER_GRADIENTS: np.cos(x_value_batch)}
    call_model.num_calls = 0

    return call_model

  def create_bad_call_model_function(self):
    def call_model(x_value_batch, call_model_args={}, expected_keys=None):
      call_model.num_calls += 1
      return {OUTPUT_LAYER_GRADIENTS: np.cos(x_value_batch)[0]}
    call_model.num_calls = 0

    return call_model

  def testBlurIGGetMask(self):
    x_steps = 4000
    # Create a call_model_function using sess and tensors.
    call_model_function = self.create_call_model_function()

    # Calculate the Blur IG attribution of the input.
    mask = self.blur_ig_instance.GetMask(
        x_value=self.x_input_val, call_model_function=call_model_function,
        call_model_args={}, max_sigma=self.max_sigma, steps=x_steps)

    # Because the baseline is blurred, all zero values should still have some 
    # attribution (introduced noise).
    self.assertEqual(np.count_nonzero(mask), mask.size)
    # Verify the result for completeness..
    # Expected (for max_sigma=10): 1.41964...
    # mask.sum (for max_sigma=10): 1.41470...
    np.testing.assert_almost_equal(mask, self.expected_val, decimal=2)
    self.assertEqual(call_model_function.num_calls, x_steps)

  def testBlurIGGetMaskBatched(self):
    x_steps = 2001
    batch_size = 100
    expected_calls = 21  #  batch size is 100, ceil(2001/100)=21
    # Create a call_model_function using sess and tensors.
    call_model_function = self.create_call_model_function()

    # Calculate the Blur IG attribution of the input.
    mask = self.blur_ig_instance.GetMask(
        x_value=self.x_input_val,
        call_model_function=call_model_function,
        call_model_args={},
        max_sigma=self.max_sigma,
        steps=x_steps,
        batch_size=batch_size)

    # Because the baseline is blurred, all zero values should still have some 
    # attribution (introduced noise).
    self.assertEqual(np.count_nonzero(mask), mask.size)
    # Verify the result for completeness..
    # Expected (for max_sigma=10): 1.72109
    # mask.sum (for max_sigma=10): 1.694904...
    np.testing.assert_almost_equal(mask, self.expected_val, decimal=2)
    self.assertEqual(call_model_function.num_calls, expected_calls)

  def testBlurIGGetMaskSingleBatch(self):
    x_steps = 499
    batch_size = 500
    expected_calls = 1  # batch size is 500, ceil(499/500)=1
    # Create a call_model_function using sess and tensors.
    call_model_function = self.create_call_model_function()

    # Calculate the Blur IG attribution of the input.
    mask = self.blur_ig_instance.GetMask(
        x_value=self.x_input_val,
        call_model_function=call_model_function,
        call_model_args={},
        max_sigma=self.max_sigma,
        steps=x_steps,
        batch_size=batch_size)

    # Because the baseline is blurred, all zero values should still have some 
    # attribution (introduced noise).
    self.assertEqual(np.count_nonzero(mask), mask.size)
    # Verify the result for completeness..
    # Expected (for max_sigma=10): 1.72109
    # mask.sum (for max_sigma=10): 1.694904...
    np.testing.assert_almost_equal(mask, self.expected_val, decimal=2)
    self.assertEqual(call_model_function.num_calls, expected_calls)

  def testBlurIGCallModelArgs(self):
    x_steps = 50
    expected_keys = [OUTPUT_LAYER_GRADIENTS]
    call_model_args = {'foo': 'bar'}
    mock_call_model = unittest.mock.MagicMock(
        return_value={OUTPUT_LAYER_GRADIENTS: [self.x_input_val]})

    self.blur_ig_instance.GetMask(
        x_value=self.x_input_val,
        call_model_function=mock_call_model,
        call_model_args=call_model_args,
        max_sigma=self.max_sigma,
        steps=x_steps)
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

  def testBlurIGGetMaskError(self):
    x_steps = 2001
    # Create a call_model_function using sess and tensors.
    call_model_function = self.create_bad_call_model_function()
    expected_error = blur_ig.SHAPE_ERROR_MESSAGE.format(
        '\\(100, 5, 5, 1\\)','\\(5, 5, 1\\)')

    with self.assertRaisesRegex(ValueError, expected_error):
      # Calculate the Blur IG attribution of the input.
      self.blur_ig_instance.GetMask(
          x_value=self.x_input_val,
          call_model_function=call_model_function,
          call_model_args={},
          max_sigma=self.max_sigma,
          steps=x_steps,
          batch_size=100)

if __name__ == '__main__':
  unittest.main()

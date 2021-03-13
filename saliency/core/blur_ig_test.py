# Copyright 2021 Google Inc. All Rights Reserved.
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

"""Tests completeness axiom, batching, and error handling for blur_ig."""
import unittest
import unittest.mock as mock

from . import blur_ig
from .base import INPUT_OUTPUT_GRADIENTS
from .base import SHAPE_ERROR_MESSAGE
import numpy as np


class BlurIgTest(unittest.TestCase):
  """To run: "python -m saliency.core.blur_ig_test" top-level saliency directory."""

  def setUp(self):
    super().setUp()
    self.max_sigma = 10
    # All black except 2 pixels near the center.
    self.x_input_val = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.7, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ],
                                dtype=float)
    self.x_input_val = self.x_input_val.reshape((5, 5, 1))
    # Calculate the value of `y` at the input.
    y_input_val = np.sin(self.x_input_val)

    # Baseline is the fully blurred version of the input.
    self.x_baseline_val = blur_ig.gaussian_blur(
        self.x_input_val, sigma=self.max_sigma)
    y_baseline_val = np.sin(self.x_baseline_val)

    # The expected BlurIG value is equal to the difference between
    # the `y` value at the input and the `y` value at the baseline. Because each
    # value is independent, we can calculate the expected blur_ig value of each.
    #
    # Expected: [[-0, -0, -0, -0, -0],
    #            [-0, 0.641, -0, -0, -0],
    #            [-0, -0, 0.838, -0, -0],
    #            [-0, -0, -0, -0, -0],
    #            [-0, -0, -0, -0, -0]
    self.expected_val = y_input_val - y_baseline_val
    self.blur_ig_instance = blur_ig.BlurIG()

  def create_call_model_function(self):
    def call_model(x_value_batch, call_model_args=None, expected_keys=None):
      call_model.num_calls += 1
      return {INPUT_OUTPUT_GRADIENTS: np.cos(x_value_batch)}
    call_model.num_calls = 0

    return call_model

  def create_bad_call_model_function(self):
    # Bad call model function, gradients do not match shape of input.
    def call_model(x_value_batch, call_model_args=None, expected_keys=None):
      call_model.num_calls += 1
      return {INPUT_OUTPUT_GRADIENTS: np.cos(x_value_batch)[0]}
    call_model.num_calls = 0

    return call_model

  def testBlurIGGetMask(self):
    """Tests that BlurIG steps are created and aggregated correctly."""
    x_steps = 4000
    # Create a call_model_function using sess and tensors.
    call_model_function = self.create_call_model_function()

    # Calculate the Blur IG attribution of the input.
    mask = self.blur_ig_instance.GetMask(
        x_value=self.x_input_val, call_model_function=call_model_function,
        call_model_args=None, max_sigma=self.max_sigma, steps=x_steps)

    # Because the baseline is blurred, all zero values should still have some
    # attribution (introduced noise).
    self.assertEqual(np.count_nonzero(mask), mask.size)
    # Verify the result (for accuracy and therefore completeness).
    np.testing.assert_almost_equal(mask, self.expected_val, decimal=2)
    self.assertEqual(call_model_function.num_calls, x_steps)

  def testBlurIGGetMaskBatched(self):
    """Tests that multiple BlurIG batches are created and aggregated correctly."""
    x_steps = 2001
    batch_size = 100
    expected_calls = 21  #  batch size is 100, ceil(2001/100)=21
    # Create a call_model_function using sess and tensors.
    call_model_function = self.create_call_model_function()

    # Calculate the Blur IG attribution of the input.
    mask = self.blur_ig_instance.GetMask(
        x_value=self.x_input_val,
        call_model_function=call_model_function,
        call_model_args=None,
        max_sigma=self.max_sigma,
        steps=x_steps,
        batch_size=batch_size)

    # Because the baseline is blurred, all zero values should still have some
    # attribution (introduced noise).
    self.assertEqual(np.count_nonzero(mask), mask.size)
    # Verify the result (for accuracy and therefore completeness).
    np.testing.assert_almost_equal(mask, self.expected_val, decimal=2)
    self.assertEqual(call_model_function.num_calls, expected_calls)

  def testBlurIGGetMaskSingleBatch(self):
    """Tests that a single BlurIG batch is created and aggregated correctly."""
    x_steps = 499
    batch_size = 500
    expected_calls = 1  # batch size is 500, ceil(499/500)=1
    # Create a call_model_function using sess and tensors.
    call_model_function = self.create_call_model_function()

    # Calculate the Blur IG attribution of the input.
    mask = self.blur_ig_instance.GetMask(
        x_value=self.x_input_val,
        call_model_function=call_model_function,
        call_model_args=None,
        max_sigma=self.max_sigma,
        steps=x_steps,
        batch_size=batch_size)

    # Because the baseline is blurred, all zero values should still have some
    # attribution (introduced noise).
    self.assertEqual(np.count_nonzero(mask), mask.size)
    # Verify the result (for accuracy and therefore completeness).
    np.testing.assert_almost_equal(mask, self.expected_val, decimal=2)
    self.assertEqual(call_model_function.num_calls, expected_calls)

  def testBlurIGCallModelArgs(self):
    """Tests that call_model_function receives correct inputs."""
    x_steps = 50
    expected_keys = [INPUT_OUTPUT_GRADIENTS]
    call_model_args = {'foo': 'bar'}
    mock_call_model = mock.MagicMock(
        return_value={INPUT_OUTPUT_GRADIENTS: [self.x_input_val]})

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
    """Tests that BlurIG errors with incorrect model outputs."""
    x_steps = 2001
    call_model_function = self.create_bad_call_model_function()
    expected_error = SHAPE_ERROR_MESSAGE[INPUT_OUTPUT_GRADIENTS].format(
        '\\(100, 5, 5, 1\\)', '\\(5, 5, 1\\)')

    # Expect error because shape of gradients returned don't match.
    with self.assertRaisesRegex(ValueError, expected_error):
      self.blur_ig_instance.GetMask(
          x_value=self.x_input_val,
          call_model_function=call_model_function,
          call_model_args=None,
          max_sigma=self.max_sigma,
          steps=x_steps,
          batch_size=100)

if __name__ == '__main__':
  unittest.main()

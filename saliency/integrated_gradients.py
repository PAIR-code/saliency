# Copyright 2017 Google Inc. All Rights Reserved.
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

"""Utilities to compute an IntegratedGradients SaliencyMask."""

import numpy as np
from .base import CallModelSaliency, OUTPUT_GRADIENTS


class IntegratedGradients(CallModelSaliency):
  """A CallModelSaliency class that implements the integrated gradients method.

  https://arxiv.org/abs/1703.01365
  """

  def GetMask(self, x_value, call_model_function, call_model_args=None,
              x_baseline=None, x_steps=25):
    """Returns a integrated gradients mask.

    Args:
      x_value: Input values to be passed to call_model function.
      call_model_function: Function that when called with an np.ndarray with
        shape equal to the input value and call_model_args, returns relevant
        outputs read from the model in the form of a dict of np.ndarrays.
        For this method, call_model_function should return the following keys:
          - 'output_gradients'
      call_model_args: (Optional) Extra parameters that are passed to the
        call_model_function.
      x_baseline: Baseline value used in integration. Defaults to 0.
      x_steps: Number of integrated steps between baseline and x.
    """
    if x_baseline is None:
      x_baseline = np.zeros_like(x_value)

    assert x_baseline.shape == x_value.shape

    x_diff = x_value - x_baseline

    total_gradients = np.zeros_like(x_value)

    for alpha in np.linspace(0, 1, x_steps):
      x_step = x_baseline + alpha * x_diff

      call_model_data = call_model_function(
          [x_step], call_model_args, expected_keys=[OUTPUT_GRADIENTS])
      total_gradients += call_model_data[OUTPUT_GRADIENTS]

    return total_gradients * x_diff / x_steps

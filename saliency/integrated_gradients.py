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

from .base import CallModelSaliency
from .base import OUTPUT_GRADIENTS
import numpy as np


class IntegratedGradients(CallModelSaliency):
  """A CallModelSaliency class that implements the integrated gradients method.

  https://arxiv.org/abs/1703.01365
  """

  def GetMask(self, x_value, call_model_function, call_model_args=None,
              x_baseline=None, x_steps=25, batch_size=1):
    """Returns a integrated gradients mask.

    Args:
      x_value: Input ndarray.
      call_model_function: A function that interfaces with a model to return
        specific data in a dictionary when given an input and other arguments.
        Expected function signature:
        - call_model_function(x_value_batch,
                              call_model_args=None,
                              expected_keys=None):
          x_value_batch - Input for the model, given as a batch (i.e. dimension
            0 is the batch dimension, dimensions 1 through n represent a single
            input).
          call_model_args - Other arguments used to call and run the model.
          expected_keys - List of keys that are expected in the output. For this
            method (Integrated Gradients), the expected keys are
            OUTPUT_GRADIENTS - Gradients of the output layer (logit/softmax)
              with respect to the input. Shape should be the same shape as
              x_value_batch.
      call_model_args: The arguments that will be passed to the call model
        function, for every call of the model.
      x_baseline: Baseline value used in integration. Defaults to 0.
      x_steps: Number of integrated steps between baseline and x.
    """
    if x_baseline is None:
      x_baseline = np.zeros_like(x_value)

    assert x_baseline.shape == x_value.shape

    x_diff = x_value - x_baseline

    total_gradients = np.zeros_like(x_value)

    x_step_batched = []
    for alpha in np.linspace(0, 1, x_steps):
      x_step = x_baseline + alpha * x_diff
      x_step_batched.append(x_step)
      if len(x_step_batched)==batch_size or alpha==1:
        call_model_data = call_model_function(
            x_step_batched, call_model_args, expected_keys=[OUTPUT_GRADIENTS])
        total_gradients += call_model_data[OUTPUT_GRADIENTS].sum(axis=0)
        x_step_batched = []

    return total_gradients * x_diff / x_steps

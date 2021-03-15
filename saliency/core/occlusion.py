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

"""Utilities to compute saliency using the occlusion method."""

from .base import CoreSaliency
from .base import OUTPUT_LAYER_VALUES
import numpy as np


class Occlusion(CoreSaliency):
  """A CoreSaliency class that computes saliency masks by occluding the image.

  This method slides a window over the image and computes how that occlusion
  affects the class score. When the class score decreases, this is positive
  evidence for the class, otherwise it is negative evidence.
  """

  expected_keys = [OUTPUT_LAYER_VALUES]

  def getY(self, x_value, call_model_function, call_model_args):
    x_value_batched = np.expand_dims(x_value, axis=0)
    data = call_model_function(
        x_value_batched,
        call_model_args=call_model_args,
        expected_keys=self.expected_keys)
    self.format_and_check_call_model_output(data,
                                x_value_batched.shape,
                                self.expected_keys)
    return data[OUTPUT_LAYER_VALUES][0]

  def GetMask(self,
              x_value,
              call_model_function,
              call_model_args=None,
              size=15,
              value=0):
    """Returns an occlusion mask.

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
            method (Occlusion), the expected keys are
            OUTPUT_LAYER_VALUES - Values of the output being
              explained (the logit/softmax value) with respect to the input.
              Outermost dimension should be the same size as x_value_batch.
      call_model_args: The arguments that will be passed to the call model
        function, for every call of the model.
      size: Height and width of the occlusion window. Default is 15.
      value: Value to repalce values inside the occlusion window with. Default
        is 0.
    """
    if len(x_value.shape) > 2:
      occlusion_window = np.zeros([size, size, x_value.shape[2]])
    else:
      occlusion_window = np.zeros([size, size])
    occlusion_window.fill(value)

    occlusion_scores = np.zeros_like(x_value)

    original_y_value = self.getY(x_value, call_model_function, call_model_args)
    for row in range(1 + x_value.shape[0] - size):
      for col in range(1 + x_value.shape[1] - size):
        x_occluded = np.array(x_value)
        if len(x_value.shape) > 2:
          x_occluded[row:row+size, col:col+size, :] = occlusion_window
        else:
          x_occluded[row:row+size, col:col+size] = occlusion_window

        y_value = self.getY(x_occluded, call_model_function, call_model_args)

        score_diff = original_y_value - y_value
        if len(x_value.shape) > 2:
          occlusion_scores[row:row+size, col:col+size, :] += score_diff
        else:
          occlusion_scores[row:row+size, col:col+size] += score_diff
    return occlusion_scores

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

"""Utilities to compute saliency using the Blur IG method."""
import math

from .base import CoreSaliency
from .base import INPUT_OUTPUT_GRADIENTS
import numpy as np
from scipy import ndimage


def gaussian_blur(image, sigma):
  """Returns Gaussian blur filtered 3d (WxHxC) image.

  Args:
    image: 3 dimensional ndarray / input image (W x H x C).
    sigma: Standard deviation for Gaussian blur kernel.
  """
  if sigma == 0:
    return image
  return ndimage.gaussian_filter(
      image, sigma=[sigma, sigma, 0], mode="constant")


class BlurIG(CoreSaliency):
  """A CoreSaliency class that implements integrated gradients along blur path.

  https://arxiv.org/abs/2004.03383

  Generates a saliency mask by computing integrated gradients for a given input
  and prediction label using a path that successively blurs the image.
  """

  expected_keys = [INPUT_OUTPUT_GRADIENTS]

  def GetMask(self,
              x_value,
              call_model_function,
              call_model_args=None,
              max_sigma=50,
              steps=100,
              grad_step=0.01,
              sqrt=False,
              batch_size=1):
    """Returns an integrated gradients mask on a blur path.

    TODO(vsubhashini): Decide if we want to restrict and find explanation
      between points of maximum information.

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
            method (Blur IG), the expected keys are
            INPUT_OUTPUT_GRADIENTS - Gradients of the output being
              explained (the logit/softmax value) with respect to the input.
              Shape should be the same shape as x_value_batch.
      call_model_args: The arguments that will be passed to the call model
        function, for every call of the model.
      max_sigma: Maximum size of the gaussian blur kernel.
      steps: Number of successive blur applications between x and fully blurred
        image (with kernel max_sigma).
      grad_step: Gaussian gradient step size.
      sqrt: Chooses square root when deciding spacing between sigma. (Full
        mathematical implication remains to be understood).
      batch_size: Maximum number of x inputs (steps along the integration path)
        that are passed to call_model_function as a batch.
    """

    if sqrt:
      sigmas = [math.sqrt(float(i)*max_sigma/float(steps)
                          ) for i in range(0, steps+1)]
    else:
      sigmas = [float(i)*max_sigma/float(steps) for i in range(0, steps+1)]
    step_vector_diff = [sigmas[i+1] - sigmas[i] for i in range(0, steps)]

    total_gradients = np.zeros_like(x_value)
    x_step_batched = []
    gaussian_gradient_batched = []
    for i in range(steps):
      x_step = gaussian_blur(x_value, sigmas[i])
      gaussian_gradient = (gaussian_blur(x_value, sigmas[i] + grad_step)
                           - x_step) / grad_step
      x_step_batched.append(x_step)
      gaussian_gradient_batched.append(gaussian_gradient)
      if len(x_step_batched) == batch_size or i == steps - 1:
        x_step_batched = np.asarray(x_step_batched)
        call_model_output = call_model_function(
            x_step_batched,
            call_model_args=call_model_args,
            expected_keys=self.expected_keys)
        self.format_and_check_call_model_output(call_model_output,
                                                x_step_batched.shape,
                                                self.expected_keys)

        tmp = (
            step_vector_diff[i] *
            np.multiply(gaussian_gradient_batched,
                        call_model_output[INPUT_OUTPUT_GRADIENTS]))
        total_gradients += tmp.sum(axis=0)
        x_step_batched = []
        gaussian_gradient_batched = []

    total_gradients *= -1.0
    return total_gradients

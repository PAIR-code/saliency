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

"""Utilities to compute Blur IG SaliencyMask.

Implementation of Integrated Gradients along the blur path.
"""
import math

from .base import GradientSaliency
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
  return ndimage.gaussian_filter(image,
                                 sigma=[sigma, sigma, 0],
                                 mode='constant')


class BlurIG(GradientSaliency):
  """A SaliencyMask class that implements integrated gradients along blur path.

  Generates a saliency mask by computing integrated gradients for a given input
  and prediction label using a path that successively blurs the image.
  TODO(vsubhashini): Add link to paper after it goes up on arxiv.
  """

  def GetMask(self, x_value, feed_dict={}, max_sigma=50, steps=100,
              grad_step=0.01, sqrt=False):
    """Returns an integrated gradients mask.

    TODO(vsubhashini): Decide if we want to restrict and find explanation
      between points of maximum information.

    Args:
      x_value: Input ndarray.
      feed_dict: Feed dictionary for the model.
      max_sigma: Maximum size of the gaussian blur kernel.
      steps: Number of successive blur applications between x and until fully
       blurred image (with kernel max_sigma).
      grad_step: Gaussian gradient step size.
      sqrt: Chooses square root when deciding spacing between sigma. (Full
            mathematical implication remains to be understood).
    """

    if sqrt:
      sigmas = [math.sqrt(float(i)*max_sigma/float(steps)
                          ) for i in range(0, steps+1)]
    else:
      sigmas = [float(i)*max_sigma/float(steps) for i in range(0, steps+1)]
    step_vector_diff = [sigmas[i+1] - sigmas[i] for i in range(0, steps)]

    total_gradients = np.zeros_like(x_value)
    for i in range(steps):
      x_step = gaussian_blur(x_value, sigmas[i])
      gaussian_gradient = (gaussian_blur(x_value, sigmas[i] + grad_step)
                           - x_step) / grad_step
      total_gradients += step_vector_diff[i] * np.multiply(
          gaussian_gradient, super(BlurIG, self).GetMask(x_step, feed_dict))

    total_gradients *= -1.0
    return total_gradients

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
from .base import GradientSaliency

class IntegratedGradients(GradientSaliency):
  """A SaliencyMask class that implements the integrated gradients method.

  https://arxiv.org/abs/1703.01365
  """

  def GetMask(self, x_value, feed_dict={}, x_baseline=None, x_steps=25):
    """Returns a integrated gradients mask.

    Args:
      x_value: input ndarray.
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

      total_gradients += super(IntegratedGradients, self).GetMask(
          x_step, feed_dict)

    return total_gradients * x_diff / x_steps

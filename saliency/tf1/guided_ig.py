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

"""Guided IG implementation for Tensorflow 1."""
from .base import TF1CoreSaliency
from ..core import guided_ig as core_guided_ig


class GuidedIG(TF1CoreSaliency):
  r"""Implements Guided IG implementation for Tensorflow 1.

  https://arxiv.org/TBD
  """

  def __init__(self, graph, session, y, x):
    super(GuidedIG, self).__init__(graph, session, y, x)
    self.core_instance = core_guided_ig.GuidedIG()

  def GetMask(self,
              x_value,
              feed_dict=None,
              x_baseline=None,
              x_steps=200,
              fraction=0.25,
              max_dist=0.02):
    """Returns Guided IG attributions.

    Args:
      x_value: Input value, not batched.
      feed_dict: (Optional) feed dictionary to pass to the session.run call.
      x_baseline: Baseline value used in integration. Defaults to 0.
      x_steps: the number of Riemann sum steps for path integral approximation.
      fraction: the fraction of features [0, 1] that should be selected and
        changed at every approximation step. E.g., value `0.25` means that 25% of
        the input features with smallest gradients are selected and changed at
        every step.
      max_dist: the relative maximum L1 distance [0, 1] that any feature can
        deviate from the straight line path. Value `0` allows no deviation and;
        therefore, corresponds to the Integrated Gradients method that is
        calculated on the straight-line path. Value `1` corresponds to the
        unbounded Guided IG method, where the path can go through any point within
        the baseline-input hyper-rectangular.
    """
    return self.core_instance.GetMask(x_value,
                                      self.call_model_function,
                                      call_model_args=feed_dict,
                                      x_baseline=x_baseline,
                                      x_steps=x_steps,
                                      fraction=fraction,
                                      max_dist=max_dist)

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

"""Utilities to compute GradientSaliency for a TF1 model."""
from .base import TF1CoreSaliency
from ..core import gradients as core_gradients


class GradientSaliency(TF1CoreSaliency):
  r"""A TF1CoreSaliency class that computes saliency masks with a gradient."""

  def __init__(self, graph, session, y, x):
    super(GradientSaliency, self).__init__(graph, session, y, x)
    self.core_instance = core_gradients.GradientSaliency()

  def GetMask(self, x_value, feed_dict=None):
    """Returns a vanilla gradient mask.

    Args:
      x_value: Input value, not batched.
      feed_dict: (Optional) feed dictionary to pass to the session.run call.
    """
    return self.core_instance.GetMask(x_value,
                                      self.call_model_function,
                                      call_model_args=feed_dict)

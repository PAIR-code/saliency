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

"""Utilities to compute saliency for a TF1 model using the occlusion method."""
from .base import TF1CoreSaliency
from ..core import occlusion as core_occlusion


class Occlusion(TF1CoreSaliency):
  r"""A TF1CoreSaleincy class that computes saliency masks using occlusion."""

  def __init__(self, graph, session, y, x):
    super(Occlusion, self).__init__(graph, session, y, x)
    self.core_instance = core_occlusion.Occlusion()

  def GetMask(self, x_value, feed_dict=None, size=15, value=0):
    """Returns an occlusion mask.

    Args:
      x_value: Input value, not batched.
      feed_dict: (Optional) feed dictionary to pass to the session.run call.
      size: Height and width of the occlusion window. Default is 15.
      value: Value to repalce values inside the occlusion window with. Default
        is 0.
    """
    return self.core_instance.GetMask(x_value,
                                      self.call_model_function,
                                      call_model_args=feed_dict,
                                      size=size,
                                      value=value)

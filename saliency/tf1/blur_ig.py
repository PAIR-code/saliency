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

"""Utilities to compute saliency for a TF1 model using the BlurIG method."""
from .base import TF1CoreSaliency
from ..core import blur_ig as core_blur_ig


class BlurIG(TF1CoreSaliency):
  """A TF1CoreSaliency class that implements IG along blur path.

  https://arxiv.org/abs/2004.03383

  Generates a saliency mask by computing integrated gradients for a given input
  and prediction label using a path that successively blurs the image.
  """

  def __init__(self, graph, session, y, x):
    super(BlurIG, self).__init__(graph, session, y, x)
    self.core_instance = core_blur_ig.BlurIG()

  def GetMask(self,
              x_value,
              feed_dict=None,
              max_sigma=50,
              steps=100,
              grad_step=0.01,
              sqrt=False,
              batch_size=1):
    """Returns a BlurIG mask.

    Args:
      x_value: Input value, not batched.
      feed_dict: (Optional) feed dictionary to pass to the session.run call.
      max_sigma: Maximum size of the gaussian blur kernel.
      steps: Number of successive blur applications between x and fully blurred
        image (with kernel max_sigma).
      grad_step: Gaussian gradient step size.
      sqrt: Chooses square root when deciding spacing between sigma. (Full
        mathematical implication remains to be understood).
      batch_size: Maximum number of x inputs (steps along the integration path)
        that are passed to sess.run as a batch.
    """
    self.validate_xy_tensor_shape(steps, batch_size)

    return self.core_instance.GetMask(x_value,
                                      self.call_model_function,
                                      call_model_args=feed_dict,
                                      max_sigma=max_sigma,
                                      steps=steps,
                                      grad_step=grad_step,
                                      sqrt=sqrt,
                                      batch_size=batch_size)

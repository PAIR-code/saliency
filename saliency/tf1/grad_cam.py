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

"""Utilities to compute saliency for a TF1 model using the GradCam method."""
from .base import TF1CoreSaliency
from ..core import grad_cam as core_grad_cam


class GradCam(TF1CoreSaliency):
  """A TF1CoreSaliency class that computes saliency masks with Grad-CAM.

  https://arxiv.org/abs/1610.02391

  Example usage (based on Examples.ipynb):

  grad_cam = GradCam(graph, sess, x=x, conv_layer=conv_layer)
  mask = grad_cam.GetMask(im,
                          should_resize = False,
                          three_dims = False)

  The Grad-CAM paper suggests using the last convolutional layer, which would
  be 'Mixed_5c' in inception_v2 and 'Mixed_7c' in inception_v3.

  """

  def __init__(self, graph, session, y, x, conv_layer):
    super(GradCam, self).__init__(graph, session, y, x, conv_layer)
    self.core_instance = core_grad_cam.GradCam()

  def GetMask(self, x_value, feed_dict=None, should_resize=True, three_dims=True):
    """Returns a GradCAM mask.

    Args:
      x_value: Input value, not batched.
      feed_dict: (Optional) feed dictionary to pass to the session.run call.
      should_resize: boolean that determines whether a low-res Grad-CAM mask
        should be upsampled to match the size of the input image
      three_dims: boolean that determines whether the grayscale mask should be
        converted into a 3D mask by copying the 2D mask value's into each color
        channel
    """
    return self.core_instance.GetMask(x_value,
                                      self.call_model_function,
                                      call_model_args=feed_dict,
                                      should_resize=should_resize,
                                      three_dims=three_dims)

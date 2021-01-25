# Copyright 2017 Ruth Fong. All Rights Reserved.
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

"""Utilities to compute SaliencyMasks."""
from .base import CallModelSaliency
from .base import CONVOLUTION_GRADIENTS
from .base import CONVOLUTION_LAYER
import numpy as np
import tensorflow.compat.v1 as tf


class GradCam(CallModelSaliency):
  """A CallModelSaliency class that computes saliency masks with Grad-CAM.

  https://arxiv.org/abs/1610.02391

  Example usage (based on Examples.ipynb):

  grad_cam = GradCam()
  mask = grad_cam.GetMask(im,
                          call_model_function,
                          call_model_args = {neuron_selector: prediction_class},
                          should_resize = False,
                          three_dims = False)

  The Grad-CAM paper suggests using the last convolutional layer, which would
  be 'Mixed_5c' in inception_v2 and 'Mixed_7c' in inception_v3.

  """

  def GetMask(self,
              x_value,
              call_model_function,
              call_model_args=None,
              should_resize=True,
              three_dims=True):
    """Returns a Grad-CAM mask.

    Modified from
    https://github.com/Ankush96/grad-cam.tensorflow/blob/master/main.py#L29-L62

    Args:
      x_value: Input value, not batched.
      feed_dict: (Optional) feed dictionary to pass to the session.run call.
      should_resize: boolean that determines whether a low-res Grad-CAM mask
        should be upsampled to match the size of the input image
      three_dims: boolean that determines whether the grayscale mask should be
        converted into a 3D mask by copying the 2D mask value's into each color
        channel
    """
    data = call_model_function(
        [x_value],
        call_model_args,
        expected_keys=[CONVOLUTION_LAYER, CONVOLUTION_GRADIENTS])

    weights = np.mean(data[CONVOLUTION_GRADIENTS], axis=(0, 1))
    grad_cam = np.zeros(data[CONVOLUTION_LAYER].shape[0:2],
                        dtype=np.float32)

    # weighted average
    for i, w in enumerate(weights):
      grad_cam += w * data[CONVOLUTION_LAYER][:, :, i]

    # pass through relu
    grad_cam = np.maximum(grad_cam, 0)

    # resize heatmap to be the same size as the input
    if should_resize:
      # values need to be [0,1] to be resized
      grad_cam = grad_cam / np.max(grad_cam)
      with tf.Graph().as_default():
        grad_cam = np.squeeze(
            tf.image.resize_bilinear(
                np.expand_dims(np.expand_dims(grad_cam, 0), 3),
                x_value.shape[:2]).eval(session=tf.Session()))

    # convert grayscale to 3-D
    if three_dims:
      grad_cam = np.expand_dims(grad_cam, axis=2)
      grad_cam = np.tile(grad_cam, [1, 1, 3])

    return grad_cam

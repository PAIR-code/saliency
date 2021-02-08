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
from skimage.transform import resize

GRADIENTS_SHAPE_ERROR_MESSAGE = ("Expected key CONVOLUTION_GRADIENTS to be the"
                      " same shape as input x_value_batch")
VALUES_SHAPE_ERROR_MESSAGE = ("Expected outermost dimension of "
                      " CONVOLUTION_LAYER to be the same as x_value_batch")


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
            method (GradCAM), the expected keys are
            CONVOLUTION_GRADIENTS - Gradients of the last convolution layer
              with respect to the input, including the batch dimension.
            CONVOLUTION_OUTPUT - Output of the last convolution layer
              for the given input, including the batch dimension.
      call_model_args: The arguments that will be passed to the call model
        function, for every call of the model.
      should_resize: boolean that determines whether a low-res Grad-CAM mask
        should be upsampled to match the size of the input image
      three_dims: boolean that determines whether the grayscale mask should be
        converted into a 3D mask by copying the 2D mask value's into each color
        channel
    """
    x_value_batched = np.array([x_value])
    data = call_model_function(
        x_value_batched,
        call_model_args,
        expected_keys=[CONVOLUTION_LAYER, CONVOLUTION_GRADIENTS])
    data[CONVOLUTION_GRADIENTS] = np.array(data[CONVOLUTION_GRADIENTS])
    data[CONVOLUTION_LAYER] = np.array(data[CONVOLUTION_LAYER])
    if (data[CONVOLUTION_GRADIENTS].shape != x_value_batched.shape):
      raise ValueError(GRADIENTS_SHAPE_ERROR_MESSAGE)
    if (data[CONVOLUTION_LAYER].shape[0] != x_value_batched.shape[0]):
      raise ValueError(VALUES_SHAPE_ERROR_MESSAGE)

    weights = np.mean(data[CONVOLUTION_GRADIENTS][0], axis=(0, 1))
    grad_cam = np.zeros(data[CONVOLUTION_LAYER][0].shape[0:2],
                        dtype=np.float32)

    # weighted average
    for i, w in enumerate(weights):
      grad_cam += w * data[CONVOLUTION_LAYER][0][:, :, i]

    # pass through relu
    grad_cam = np.maximum(grad_cam, 0)

    # resize heatmap to be the same size as the input
    if should_resize:
      grad_cam = grad_cam / np.max(grad_cam)
      grad_cam = resize(grad_cam, x_value.shape[:2])

    # convert grayscale to 3-D
    if three_dims:
      grad_cam = np.expand_dims(grad_cam, axis=2)
      grad_cam = np.tile(grad_cam, [1, 1, 3])

    return grad_cam

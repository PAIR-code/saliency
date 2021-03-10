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

"""Utilities to compute saliency using the GradCam method."""
from .base import CONVOLUTION_LAYER_VALUES
from .base import CONVOLUTION_OUTPUT_GRADIENTS
from .base import CoreSaliency
import numpy as np
from skimage.transform import resize


class GradCam(CoreSaliency):
  """A CoreSaliency class that computes saliency masks with Grad-CAM.

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

  expected_keys = [CONVOLUTION_LAYER_VALUES, CONVOLUTION_OUTPUT_GRADIENTS]

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
            CONVOLUTION_LAYER_VALUES - Output of the last convolution layer
              for the given input, including the batch dimension.
            CONVOLUTION_OUTPUT_GRADIENTS - Gradients of the output being
              explained (the logit/softmax value) with respect to the last
              convolution layer, including the batch dimension.
      call_model_args: The arguments that will be passed to the call model
        function, for every call of the model.
      should_resize: boolean that determines whether a low-res Grad-CAM mask
        should be upsampled to match the size of the input image
      three_dims: boolean that determines whether the grayscale mask should be
        converted into a 3D mask by copying the 2D mask value's into each color
        channel
    """
    x_value_batched = np.expand_dims(x_value, axis=0)
    data = call_model_function(x_value_batched,
        call_model_args=call_model_args,
        expected_keys=self.expected_keys)
    self.format_and_check_call_model_output(data, x_value_batched.shape, self.expected_keys)

    weights = np.mean(data[CONVOLUTION_OUTPUT_GRADIENTS][0], axis=(0, 1))
    grad_cam = np.zeros(data[CONVOLUTION_LAYER_VALUES][0].shape[0:2],
                        dtype=np.float32)

    # weighted average
    for i, w in enumerate(weights):
      grad_cam += w * data[CONVOLUTION_LAYER_VALUES][0][:, :, i]

    # pass through relu
    grad_cam = np.maximum(grad_cam, 0)

    # resize heatmap to be the same size as the input
    if should_resize:
      if np.max(grad_cam) > 0:
        grad_cam = grad_cam / np.max(grad_cam)
      grad_cam = resize(grad_cam, x_value.shape[:2])

    # convert grayscale to 3-D
    if three_dims:
      grad_cam = np.expand_dims(grad_cam, axis=2)
      grad_cam = np.tile(grad_cam, [1, 1, 3])

    return grad_cam

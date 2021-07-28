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

"""Utilities to compute saliency for a TF1 model using the XRAI method."""
from .base import TF1CoreSaliency
from ..core import xrai as core_xrai

XRAIParameters = core_xrai.XRAIParameters


class XRAI(TF1CoreSaliency):
  r"""A TF1CoreSaliency class that computes XRAI."""

  def __init__(self, graph, session, y, x):
    super(XRAI, self).__init__(graph, session, y, x)
    self.core_instance = core_xrai.XRAI()

  def GetMask(self,
              x_value,
              feed_dict=None,
              baselines=None,
              segments=None,
              base_attribution=None,
              batch_size=1,
              extra_parameters=None):
    """Applies XRAI method on an input image and returns the result saliency heatmap.


    Args:
        x_value: Input ndarray, not batched.
        feed_dict: feed dictionary to pass to the TF session.run() call.
          Defaults to {}.
        baselines: a list of baselines to use for calculating
          Integrated Gradients attribution. Every baseline in
          the list should have the same dimensions as the
          input. If the value is not set then the algorithm
          will make the best effort to select default
          baselines. Defaults to None.
        segments: the list of precalculated image segments that should
          be passed to XRAI. Each element of the list is an
          [N,M] boolean array, where NxM are the image
          dimensions. Each elemeent on the list contains exactly the
          mask that corresponds to one segment. If the value is None,
          Felzenszwalb's segmentation algorithm will be applied.
          Defaults to None.
        base_attribution: an optional pre-calculated base attribution that XRAI
          should use. The shape of the parameter should match
          the shape of `x_value`. If the value is None, the
          method calculates Integrated Gradients attribution and
          uses it.
        batch_size: Maximum number of x inputs (steps along the integration
          path) that are passed to sess.run as a batch.
        extra_parameters: an XRAIParameters object that specifies
          additional parameters for the XRAI saliency
          method. If it is None, an XRAIParameters object
          will be created with default parameters. See
          XRAIParameters for more details.

    Raises:
        ValueError: If algorithm type is unknown (not full or fast).
                    If the shape of `base_attribution` dosn't match the shape of
                      `x_value`.
                    If call_model_function cannot return INPUT_OUTPUT_GRADIENTS.

    Returns:
        np.ndarray: A numpy array that contains the saliency heatmap.


    TODO(tolgab) Add output_selector functionality from XRAI API doc
    """
    if extra_parameters is None:
      x_steps = XRAIParameters().steps
    else:
      x_steps = extra_parameters.steps
    self.validate_xy_tensor_shape(x_steps, batch_size)

    return self.core_instance.GetMask(
        x_value,
        call_model_function=self.call_model_function,
        call_model_args=feed_dict,
        baselines=baselines,
        segments=segments,
        base_attribution=base_attribution,
        batch_size=batch_size,
        extra_parameters=extra_parameters)

  def GetMaskWithDetails(self,
                         x_value,
                         feed_dict=None,
                         baselines=None,
                         segments=None,
                         base_attribution=None,
                         batch_size=1,
                         extra_parameters=None):
    """Applies XRAI method on an input image and returns the result saliency heatmap along with other detailed information.

    Args:
        x_value: Input ndarray, not batched.
        feed_dict: feed dictionary to pass to the TF session.run() call.
          Defaults to {}.
        baselines: a list of baselines to use for calculating
          Integrated Gradients attribution. Every baseline in
          the list should have the same dimensions as the
          input. If the value is not set then the algorithm
          will make the best effort to select default
          baselines. Defaults to None.
        segments: the list of precalculated image segments that should
          be passed to XRAI. Each element of the list is an
          [N,M] boolean array, where NxM are the image
          dimensions. Each elemeent on the list contains exactly the
          mask that corresponds to one segment. If the value is None,
          Felzenszwalb's segmentation algorithm will be applied.
          Defaults to None.
        base_attribution: an optional pre-calculated base attribution that XRAI
          should use. The shape of the parameter should match
          the shape of `x_value`. If the value is None, the
          method calculates Integrated Gradients attribution and
          uses it.
        batch_size: Maximum number of x inputs (steps along the integration
          path) that are passed to sess.run as a batch.
        extra_parameters: an XRAIParameters object that specifies
          additional parameters for the XRAI saliency
          method. If it is None, an XRAIParameters object
          will be created with default parameters. See
          XRAIParameters for more details.

    Raises:
        ValueError: If algorithm type is unknown (not full or fast).
                    If the shape of `base_attribution` dosn't match the shape of
                      `x_value`.
                    If call_model_function cannot return INPUT_OUTPUT_GRADIENTS.
    Returns:
        XRAIOutput: an object that contains the output of the XRAI algorithm.
    TODO(tolgab) Add output_selector functionality from XRAI API doc
    """
    if extra_parameters is None:
      x_steps = XRAIParameters().steps
    else:
      x_steps = extra_parameters.steps
    self.validate_xy_tensor_shape(x_steps, batch_size)

    return self.core_instance.GetMaskWithDetails(
        x_value,
        call_model_function=self.call_model_function,
        call_model_args=feed_dict,
        baselines=baselines,
        segments=segments,
        base_attribution=base_attribution,
        batch_size=batch_size,
        extra_parameters=extra_parameters)

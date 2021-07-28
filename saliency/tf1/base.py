# Copyright 2021 Google Inc. All Rights Reserved.
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
from . import utils
import numpy as np


X_SHAPE_ERROR_MESSAGE = ('Unexpected outermost dimension for x tensor. Expected'
                         ' {} for outermost dimension, actual {}')
Y_SHAPE_ERROR_MESSAGE = ('Unexpected shape for y tensor. Expected shape {},'
                         ' actual {}')


class TF1Saliency(object):
  """Base class for TF saliency masks. Alone, this class doesn't do anything."""

  def __init__(self, graph, session, y, x):
    """TF1-specific class to create saliency masks.

    Args:
      graph: The TensorFlow graph to evaluate masks on.
      session: The current TensorFlow session.
      y: The output tensor to compute the SaliencyMask against. This tensor
          shape should be (None,), (batch_size), or () if not batching inputs.
      x: The input tensor to compute the SaliencyMask against. The outer
          dimension should be the batch size.
    """

    self.graph = graph
    self.session = session
    self.y = y
    self.x = x

  def GetMask(self, x_value, feed_dict=None):
    """Returns an unsmoothed mask.

    Args:
      x_value: Input value, not batched.
      feed_dict: (Optional) feed dictionary to pass to the session.run call.
    """
    raise NotImplementedError('A derived class should implemented GetMask()')

  def GetSmoothedMask(self,
                      x_value,
                      feed_dict=None,
                      stdev_spread=.15,
                      nsamples=25,
                      magnitude=True,
                      **kwargs):
    """Returns a mask that is smoothed with the SmoothGrad method.

    Args:
      x_value: Input value, not batched.
      feed_dict: (Optional) feed dictionary to pass to the session.run call.
      stdev_spread: Amount of noise to add to the input, as fraction of the
        total spread (x_max - x_min). Defaults to 15%.
      nsamples: Number of samples to average across to get the smooth gradient.
      magnitude: If true, computes the sum of squares of gradients instead of
        just the sum. Defaults to true.
      **kwargs: Additional keyword arguments to be passed to GetMask.
    """
    stdev = stdev_spread * (np.max(x_value) - np.min(x_value))

    total_gradients = np.zeros_like(x_value)
    for _ in range(nsamples):
      noise = np.random.normal(0, stdev, x_value.shape)
      x_plus_noise = x_value + noise
      grad = self.GetMask(x_plus_noise, feed_dict, **kwargs)
      if magnitude:
        total_gradients += (grad * grad)
      else:
        total_gradients += grad

    return total_gradients / nsamples


class TF1CoreSaliency(TF1Saliency):
  """Base class for TF1Saliency methods that use CoreSaliency methods.

  Alone, this class doesn't do anything.
  """

  def __init__(self, graph, session, y, x, conv_layer=None):
    """TF1-specific class to create saliency masks using CoreSaliency.

    Args:
      graph: The TensorFlow graph to evaluate masks on.
      session: The current TensorFlow session.
      y: The output tensor to compute the SaliencyMask against. This tensor
          shape should be (None,), (batch_size), or () if not batching inputs.
      x: The input tensor to compute the SaliencyMask against. The outer
          dimension should be the batch size.
      conv_layer: The convolution layer tensor of the model. The outer
          dimension should be the batch size.
    """

    super(TF1CoreSaliency, self).__init__(graph, session, y, x)
    self.call_model_function = utils.create_tf1_call_model_function(
        graph, session, y, x, conv_layer)

  def validate_xy_tensor_shape(self, x_steps=1, batch_size=1):
    """Validates the shapes of x and y tensors with respect to batch_size.

    Args:
      x_steps: Number of steps computed as part of the full method.
      batch_size: Maximum batch_size to split up the x_steps.

    Raises:
        ValueError: If x tensor shape is incompatible with batching parameters.
                    If y tensor shape is incompatible with batching parameters.
    """
    batch_remainder = x_steps % batch_size
    if batch_remainder==0:
      target_size = batch_size
    else:
      target_size = None

    # y should be shape (None,), (target_size), or () if target_size=1
    y_size = 1
    y_none_dim = False
    y_shape = self.y.get_shape().as_list()
    for shape in y_shape:
      if shape is None:
        y_none_dim = True
      else:
        y_size *= shape
    y_size_tuple = (y_size, y_none_dim)
    if y_size_tuple not in [(1, True), (target_size, False)]:
      expected = '[None]'
      if target_size is not None:
        expected += ' or tensor with size {}'.format(target_size)
      raise ValueError(Y_SHAPE_ERROR_MESSAGE.format(expected, y_shape))

    # x.shape[0] should be shape None or target_size
    x_outer_size = self.x.get_shape().as_list()[0]
    if x_outer_size not in [None, target_size]:
      expected = 'None'
      if target_size is not None:
        expected += ' or {}'.format(target_size)
      raise ValueError(X_SHAPE_ERROR_MESSAGE.format(expected, x_outer_size))

  def GetMask(self, x_value, feed_dict=None):
    """Returns an unsmoothed mask.

    Args:
      x_value: Input value, not batched.
      feed_dict: (Optional) feed dictionary to pass to the session.run call.
    """
    raise NotImplementedError('A derived class should implemented GetMask()')

  def GetSmoothedMask(self,
                      x_value,
                      feed_dict=None,
                      stdev_spread=.15,
                      nsamples=25,
                      magnitude=True,
                      **kwargs):
    """Returns a mask that is smoothed with the SmoothGrad method.

    Args:
      x_value: Input value, not batched.
      feed_dict: (Optional) feed dictionary to pass to the session.run call.
      stdev_spread: Amount of noise to add to the input, as fraction of the
        total spread (x_max - x_min). Defaults to 15%.
      nsamples: Number of samples to average across to get the smooth gradient.
      magnitude: If true, computes the sum of squares of gradients instead of
        just the sum. Defaults to true.
      **kwargs: Additional keyword arguments to be passed to GetMask.
    """
    stdev = stdev_spread * (np.max(x_value) - np.min(x_value))

    total_gradients = np.zeros_like(x_value)
    for _ in range(nsamples):
      noise = np.random.normal(0, stdev, x_value.shape)
      x_plus_noise = x_value + noise
      grad = self.GetMask(x_plus_noise, feed_dict, **kwargs)
      if magnitude:
        total_gradients += (grad * grad)
      else:
        total_gradients += grad

    return total_gradients / nsamples

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


class TF1Saliency(object):
  """Base class for TF saliency masks. Alone, this class doesn't do anything."""

  def __init__(self, graph, session, y=None, x=None):
    """TF1-specific class to create saliency masks.

    Args:
      graph: The TensorFlow graph to evaluate masks on.
      session: The current TensorFlow session.
      y: The output tensor to compute the SaliencyMask against. This tensor
          should be of size 1.
      x: The input tensor to compute the SaliencyMask against. The outer
          dimension should be the batch size.
    """

    self.graph = graph
    self.session = session
    self.y = y
    self.x = x

  def GetMask(self, x_value, feed_dict={}):
    """Returns an unsmoothed mask.

    Args:
      x_value: Input value, not batched.
      feed_dict: (Optional) feed dictionary to pass to the session.run call.
    """
    raise NotImplementedError('A derived class should implemented GetMask()')

  def GetSmoothedMask(self,
                      x_value,
                      feed_dict={},
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

  def __init__(self, graph, session, y=None, x=None, conv_layer=None):
    """TF1-specific class to create saliency masks using CoreSaliency.

    Args:
      graph: The TensorFlow graph to evaluate masks on.
      session: The current TensorFlow session.
      y: The output tensor to compute the SaliencyMask against. This tensor
          should be of size 1.
      x: The input tensor to compute the SaliencyMask against. The outer
          dimension should be the batch size.
      conv_layer: The convolution layer tensor of the model. The outer
          dimension should be the batch size.
    """

    super(TF1CoreSaliency, self).__init__(graph, session, y, x)
    self.call_model_function = utils.create_tf1_call_model_function(
        graph, session, y, x, conv_layer)

  def GetMask(self, x_value, feed_dict={}):
    """Returns an unsmoothed mask.

    Args:
      x_value: Input value, not batched.
      feed_dict: (Optional) feed dictionary to pass to the session.run call.
    """
    raise NotImplementedError('A derived class should implemented GetMask()')

  def GetSmoothedMask(self,
                      x_value,
                      feed_dict={},
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

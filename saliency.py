# Copyright 2017 Google Inc. All Rights Reserved.
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
import numpy as np
import tensorflow as tf


class SaliencyMask(object):
  """Base class for saliency masks. Alone, this class doesn't do anything."""
  def __init__(self, graph, session, y, x):
    """Constructs a SaliencyMask.

    Args:
      graph: The TensorFlow graph to evaluate masks on.
      session: The current TensorFlow session.
      y: The output tensor to compute the SaliencyMask against. This tensor
          should be of size 1.
      x: The input tensor to compute the SaliencyMask against. The outer
          dimension should be the batch size.
    """

    # y must be of size one, otherwise the gradient we get from tf.gradients
    # will be summed over all ys.
    size = 1
    for shape in y.shape:
      size *= shape
    assert size == 1

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
    pass

  def GetSmoothedMask(
      self, x_value, feed_dict={}, stdev_spread=.2, nsamples=50):
    """Returns a mask that is smoothed with the SmoothGrad method.

    Args:
      x_value: Input value, not batched.
      feed_dict: (Optional) feed dictionary to pass to the session.run call.
    """
    stdev = stdev_spread * (np.max(x_value) - np.min(x_value))

    total_gradients = np.zeros_like(x_value)
    for i in range(nsamples):
      noise = np.random.normal(0, stdev, x_value.shape)
      x_plus_noise = x_value + noise

      total_gradients += self.GetMask(x_plus_noise, feed_dict)

    return total_gradients / nsamples

class GradientSaliency(SaliencyMask):
  r"""A SaliencyMask class that computes saliency masks with a gradient."""

  def __init__(self, graph, session, y, x):
    super(GradientSaliency, self).__init__(graph, session, y, x)
    self.gradients_node = tf.gradients(y, x)[0]

  def GetMask(self, x_value, feed_dict={}):
    """Returns a vanilla gradient mask.

    Args:
      x_value: Input value, not batched.
      feed_dict: (Optional) feed dictionary to pass to the session.run call.
    """
    feed_dict[self.x] = [x_value]
    return self.session.run(self.gradients_node, feed_dict=feed_dict)[0]

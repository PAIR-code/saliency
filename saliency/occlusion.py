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

"""Utilities to compute an Occlusion SaliencyMask."""

import numpy as np

from .base import SaliencyMask

class Occlusion(SaliencyMask):
  """A SaliencyMask class that computes saliency masks by occluding the image.

  This method slides a window over the image and computes how that occlusion
  affects the class score. When the class score decreases, this is positive
  evidence for the class, otherwise it is negative evidence.
  """

  def __init__(self, graph, session, y, x):
    super(Occlusion, self).__init__(graph, session, y, x)

  def GetMask(self, x_value, feed_dict = {}, size = 15, value = 0):
    """Returns an occlusion mask."""
    occlusion_window = np.array([size, size, x_value.shape[2]])
    occlusion_window.fill(value)

    occlusion_scores = np.zeros_like(x_value)

    feed_dict[self.x] = [x_value]
    original_y_value = self.session.run(self.y, feed_dict=feed_dict)

    for row in range(x_value.shape[0] - size):
      for col in range(x_value.shape[1] - size):
        x_occluded = np.array(x_value)

        x_occluded[row:row+size, col:col+size, :] = occlusion_window

        feed_dict[self.x] = [x_occluded]
        y_value = self.session.run(self.y, feed_dict=feed_dict)

        score_diff = original_y_value - y_value
        occlusion_scores[row:row+size, col:col+size, :] += score_diff
    return occlusion_scores

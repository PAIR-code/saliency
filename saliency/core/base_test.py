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

"""Tests accuracy for CoreSaliency class."""
import unittest
import unittest.mock as mock

from . import base
import numpy as np


class CoreSaliencyTest(unittest.TestCase):
  """To run: "python -m saliency.core.base_test" from the top-level directory."""

  def testGetSmoothedMaskDtype(self):
    """Tests that GetSmoothedMask works on int-based inputs."""
    n_samples = 20
    x_input = np.array([3, 2, 1])
    return_value = np.array([.5, .2, .1])
    mock_get_mask = mock.MagicMock(return_value=return_value)
    core_instance = base.CoreSaliency()
    core_instance.GetMask = mock_get_mask

    smoothed_mask = core_instance.GetSmoothedMask(x_input, None,
                                                  nsamples=n_samples)

    self.assertEqual(len(mock_get_mask.mock_calls), n_samples)

if __name__ == '__main__':
  unittest.main()

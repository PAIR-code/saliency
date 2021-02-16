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

from ..core import integrated_gradients as core_integrated_gradients
from .base import TF1CoreSaliency

class IntegratedGradients(TF1CoreSaliency):
  r"""A TF1CoreSaliency class that computes saliency using Integrated Gradients.

  https://arxiv.org/abs/1703.01365"""

  def __init__(self, graph, session, y, x):
    super(IntegratedGradients, self).__init__(graph, session, y, x)
    self.core_instance = core_integrated_gradients.IntegratedGradients()

  def GetMask(self, x_value, feed_dict={},
              x_baseline=None, x_steps=25, batch_size=1):
    """Returns an integrated gradients mask.

    Args:
      x_value: Input value, not batched.
      feed_dict: (Optional) feed dictionary to pass to the session.run call.
      x_baseline: Baseline value used in integration. Defaults to 0.
      x_steps: Number of integrated steps between baseline and x.
      batch_size: Maximum number of x inputs that are passed to session.run call
        as a batch.
    """
    return self.core_instance.GetMask(x_value, 
        self.call_model_function,
        call_model_args=feed_dict,
        x_baseline=x_baseline,
        x_steps=x_steps,
        batch_size=batch_size)
# Copyright 2018 Google Inc. All Rights Reserved.
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
# ==============================================================================
import unittest

from . import xrai
import numpy as np
import tensorflow.compat.v1 as tf

INPUT_HEIGHT_WIDTH = 5  # width and height of input images in pixels


class XRAITest(unittest.TestCase):
  """To run: "python -m saliency.grad_cam_test" from top-level saliency directory."""

  def setUp(self):
      super().setUp()
      with tf.Graph().as_default() as graph:
        x = tf.placeholder(shape=[None, 3], dtype=tf.float32)
        contrib = [5 * x[:, 0], x[:, 1] * x[:, 1], tf.sin(x[:, 2])]
        y = contrib[0] + contrib[1] + contrib[2]
        sess = tf.Session(graph=graph)

        # Calculate the integrated gradients attribution of the input.
        self.xrai_instance = xrai.XRAI(graph,
                                      sess,
                                      y,
                                      x)

  def testXraiGetMaskArgs(self):
    x_value = [3,2,1]
    feed_dict = {'foo':'bar'}
    baselines = 'baz'
    segments = 'baq'
    base_attribution = [1,2,3]
    batch_size = 9
    extra_parameters = {'xr' : 'ai'}
    self.xrai_instance.core_instance.GetMask = unittest.mock.MagicMock()
    mock = self.xrai_instance.core_instance.GetMask

    self.xrai_instance.GetMask(x_value=x_value,
                                feed_dict=feed_dict, 
                                baselines=baselines, 
                                segments=segments, 
                                base_attribution=base_attribution, 
                                batch_size=batch_size, 
                                extra_parameters=extra_parameters)

    self.assertEqual(mock.call_args_list, 
                    [unittest.mock.call(x_value, 
                    call_model_function=self.xrai_instance.call_model_function, 
                    call_model_args=feed_dict, 
                    baselines=baselines, 
                    segments=segments, 
                    base_attribution=base_attribution, 
                    batch_size=batch_size, 
                    extra_parameters=extra_parameters)])

  def testXraiGetMaskWithDetailsArgs(self):
    x_value = [3,2,1]
    feed_dict = {'foo':'bar'}
    baselines = 'baz'
    segments = 'baq'
    base_attribution = [1,2,3]
    batch_size = 9
    extra_parameters = {'xr' : 'ai'}
    self.xrai_instance.core_instance.GetMaskWithDetails = unittest.mock.MagicMock()
    mock = self.xrai_instance.core_instance.GetMaskWithDetails

    self.xrai_instance.GetMaskWithDetails(x_value=x_value,
                                feed_dict=feed_dict, 
                                baselines=baselines, 
                                segments=segments, 
                                base_attribution=base_attribution, 
                                batch_size=batch_size, 
                                extra_parameters=extra_parameters)

    self.assertEqual(mock.call_args_list, 
                    [unittest.mock.call(x_value, 
                    call_model_function=self.xrai_instance.call_model_function, 
                    call_model_args=feed_dict, 
                    baselines=baselines, 
                    segments=segments, 
                    base_attribution=base_attribution, 
                    batch_size=batch_size, 
                    extra_parameters=extra_parameters)])

if __name__ == '__main__':
  unittest.main()


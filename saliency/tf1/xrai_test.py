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

"""Tests accuracy and correct TF1 usage for xrai."""
import unittest
import unittest.mock as mock

from . import xrai
import numpy as np
import tensorflow.compat.v1 as tf


class XRAITest(unittest.TestCase):
  """To run: "python -m saliency.tf1.xrai_test" from top-level saliency directory."""

  def setUp(self):
    super().setUp()
    self.graph = tf.Graph()
    with self.graph.as_default():
      x = tf.placeholder(shape=[None, 3], dtype=tf.float32)
      contrib = [5 * x[:, 0], x[:, 1] * x[:, 1], 2 * x[:, 2]]
      y = contrib[0] + contrib[1] + contrib[2]
      sess = tf.Session(graph=self.graph)
      self.sess_spy = mock.MagicMock(wraps=sess)

      # Calculate the integrated gradients attribution of the input.
      self.x = x
      self.y = y
      self.x_indexed = x[0]
      self.y_indexed = x[0]

  def testXraiGetMaskArgs(self):
    """Test that the CoreSaliency GetMask method is called correctly."""
    self.xrai_instance = xrai.XRAI(self.graph, self.sess_spy, self.y, self.x)
    x_value = [3, 2, 1]
    feed_dict = {'foo': 'bar'}
    baselines = 'baz'
    segments = 'baq'
    base_attribution = [1, 2, 3]
    batch_size = 9
    extra_parameters = xrai.XRAIParameters(steps=900)
    self.xrai_instance.core_instance.GetMask = mock.MagicMock()
    mock_function = self.xrai_instance.core_instance.GetMask

    self.xrai_instance.GetMask(
        x_value=x_value,
        feed_dict=feed_dict,
        baselines=baselines,
        segments=segments,
        base_attribution=base_attribution,
        batch_size=batch_size,
        extra_parameters=extra_parameters)

    self.assertEqual(mock_function.call_args_list, [
        mock.call(
            x_value,
            call_model_function=self.xrai_instance.call_model_function,
            call_model_args=feed_dict,
            baselines=baselines,
            segments=segments,
            base_attribution=base_attribution,
            batch_size=batch_size,
            extra_parameters=extra_parameters)
    ])

  def testXraiGetMaskWithDetailsArgs(self):
    """Test that the CoreSaliency GetMaskWithDetails method is called correctly."""
    self.xrai_instance = xrai.XRAI(self.graph, self.sess_spy, self.y, self.x)
    x_value = [3, 2, 1]
    feed_dict = {'foo': 'bar'}
    baselines = 'baz'
    segments = 'baq'
    base_attribution = [1, 2, 3]
    batch_size = 9
    extra_parameters = xrai.XRAIParameters(steps=900)
    core_instance = self.xrai_instance.core_instance
    core_instance.GetMaskWithDetails = mock.MagicMock()
    mock_function = self.xrai_instance.core_instance.GetMaskWithDetails

    self.xrai_instance.GetMaskWithDetails(
        x_value=x_value,
        feed_dict=feed_dict,
        baselines=baselines,
        segments=segments,
        base_attribution=base_attribution,
        batch_size=batch_size,
        extra_parameters=extra_parameters)

    self.assertEqual(mock_function.call_args_list, [
        mock.call(
            x_value,
            call_model_function=self.xrai_instance.call_model_function,
            call_model_args=feed_dict,
            baselines=baselines,
            segments=segments,
            base_attribution=base_attribution,
            batch_size=batch_size,
            extra_parameters=extra_parameters)
    ])

  def testXraiFunctional(self):
    """Test that the model is called and used to calculate the XRAI mask."""
    self.xrai_instance = xrai.XRAI(self.graph, self.sess_spy, self.y, self.x)
    self.xrai_instance.validate_xy_tensor_shape = mock.MagicMock()
    feed_dict = {}
    baselines = np.array([[0, 0, 3], [0, 0, 0]])
    segments = [[True, True, False], [False, False, True]]
    batch_size = 9
    steps = 100
    extra_parameters = xrai.XRAIParameters(steps=steps,
                                           return_ig_attributions=True,
                                           return_xrai_segments=True)
    # Reducing min_pixel_diff lets us split up this very small 1x3 "image"
    extra_parameters.experimental_params['min_pixel_diff'] = 0
    expected_segments = np.array([1, 1, 2])
    x_value = np.array([3, 2, 3])
    # equation is 5x + y^2 + 2z, but first baseline has z_baseline=z_input
    expected_ig = np.array(
        [[x_value[0] * 5, x_value[1] * x_value[1], 0],
         [x_value[0] * 5, x_value[1] * x_value[1], x_value[2] * 2]])
    expected_mask = np.mean(expected_ig, axis=0)
    # segment masks have the first two inputs together
    expected_mask[0:2] = np.mean(expected_mask[0:2])  # [9.5, 9.5, 3]
    expected_calls = 24  # batch size is 9, 2*ceil(100/9)=24
    expected_validate_args = (steps, batch_size)


    mask_details = self.xrai_instance.GetMaskWithDetails(
        x_value=x_value,
        feed_dict=feed_dict,
        baselines=baselines,
        segments=segments,
        batch_size=batch_size,
        extra_parameters=extra_parameters)
    validate_args = self.xrai_instance.validate_xy_tensor_shape.call_args[0]

    # Verify the result.
    self.assertEqual(validate_args, expected_validate_args)
    np.testing.assert_almost_equal(mask_details.attribution_mask,
                                   expected_mask, decimal=4)
    np.testing.assert_almost_equal(mask_details.ig_attribution,
                                   expected_ig, decimal=4)
    np.testing.assert_almost_equal(mask_details.segments,
                                   expected_segments, decimal=4)
    np.testing.assert_almost_equal(mask_details.baselines,
                                   baselines, decimal=4)
    self.assertEqual(self.sess_spy.run.call_count, expected_calls)

if __name__ == '__main__':
  unittest.main()

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

"""Tests input and output tensor validation for TF1CoreSaliency classes."""
import unittest
from . import base
import numpy as np
import tensorflow.compat.v1 as tf


class BaseTest(unittest.TestCase):
  """To run: "python -m saliency.tf1.integrated_gradients_test" from the top-level directory."""

  def setUp(self):
    super().setUp()
    tf.enable_v2_tensorshape()
    self.graph = tf.Graph()
    with self.graph.as_default():
      x = tf.placeholder(shape=[None, 3], dtype=tf.float32)
      contrib = [5 * x[:, 0], x[:, 1] * x[:, 1], tf.sin(x[:, 2])]
      self.x = x
      self.y = (contrib[0] + contrib[1] + contrib[2])
      self.x_indexed = self.x[0]
      self.y_indexed = self.y[0]
      self.sess = tf.Session(graph=self.graph)
      self.x_input_val = np.array([1.0, 2.0, 3.0], dtype=float)

  def testValidateTensorsBatched(self):
    """Tests that now error is thrown when y tensor is a single value."""
    x_steps = 1000
    batch_size = 9
    base_instance = base.TF1CoreSaliency(self.graph,
                                         self.sess,
                                         self.y,
                                         self.x)
    
    # Check validate doesn't throw any ValueError
    base_instance.validate_xy_tensor_shape(x_steps, batch_size)

  def testValidateTensorsBatchedV1Shape(self):
    """Tests that now error is thrown when y tensor is a single value."""
    x_steps = 1000
    batch_size = 9
    tf.disable_v2_tensorshape()
    base_instance = base.TF1CoreSaliency(self.graph,
                                         self.sess,
                                         self.y,
                                         self.x)
    
    # Check validate doesn't throw any ValueError
    base_instance.validate_xy_tensor_shape(x_steps, batch_size)

  def testValidateTensorsErrorX(self):
    """Tests that an error is thrown when x tensor is set up incorrectly."""
    x_steps = 1000
    batch_size = 5
    base_instance = base.TF1CoreSaliency(self.graph,
                                         self.sess,
                                         self.y,
                                         self.x_indexed)
    expected_error = base.X_SHAPE_ERROR_MESSAGE.format(
        'None or 5', '3')

    with self.assertRaisesRegex(ValueError, expected_error):
      base_instance.validate_xy_tensor_shape(x_steps, batch_size)

  def testValidateTensorsErrorXV1Shape(self):
    """Tests that an error is thrown when x tensor is set up incorrectly."""
    x_steps = 1000
    batch_size = 5
    tf.disable_v2_tensorshape()
    base_instance = base.TF1CoreSaliency(self.graph,
                                         self.sess,
                                         self.y,
                                         self.x_indexed)
    expected_error = base.X_SHAPE_ERROR_MESSAGE.format(
        'None or 5', '3')

    with self.assertRaisesRegex(ValueError, expected_error):
      base_instance.validate_xy_tensor_shape(x_steps, batch_size)

  def testValidateTensorsErrorY(self):
    """Tests that an error is thrown when x tensor is set up incorrectly."""
    x_steps = 1000
    batch_size = 9
    base_instance = base.TF1CoreSaliency(self.graph,
                                         self.sess,
                                         self.y_indexed,
                                         self.x)
    expected_error = base.Y_SHAPE_ERROR_MESSAGE.format(
        '\\[None\\]', '\\[\\]')

    with self.assertRaisesRegex(ValueError, expected_error):
      base_instance.validate_xy_tensor_shape(x_steps, batch_size)

  def testValidateTensorsErrorYV1Shape(self):
    """Tests that an error is thrown when x tensor is set up incorrectly."""
    x_steps = 1000
    batch_size = 9
    tf.disable_v2_tensorshape()
    base_instance = base.TF1CoreSaliency(self.graph,
                                         self.sess,
                                         self.y_indexed,
                                         self.x)
    expected_error = base.Y_SHAPE_ERROR_MESSAGE.format(
        '\\[None\\]', '\\[\\]')

    with self.assertRaisesRegex(ValueError, expected_error):
      base_instance.validate_xy_tensor_shape(x_steps, batch_size)

  def testValidateTensorsSingleY(self):
    """Tests that now error is thrown when y tensor is a single value."""
    x_steps = 1000
    batch_size = 1
    base_instance = base.TF1CoreSaliency(self.graph,
                                         self.sess,
                                         self.y_indexed,
                                         self.x)
    
    # Check validate doesn't throw any ValueError
    base_instance.validate_xy_tensor_shape(x_steps, batch_size)

  def testValidateTensorsSingleYV1Shape(self):
    """Tests that now error is thrown when y tensor is a single value."""
    x_steps = 1000
    batch_size = 1
    tf.disable_v2_tensorshape()
    base_instance = base.TF1CoreSaliency(self.graph,
                                         self.sess,
                                         self.y_indexed,
                                         self.x)
    
    # Check validate doesn't throw any ValueError
    base_instance.validate_xy_tensor_shape(x_steps, batch_size)


if __name__ == '__main__':
  unittest.main()
  
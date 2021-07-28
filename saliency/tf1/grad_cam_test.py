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

"""Tests accuracy and correct TF1 usage for grad_cam."""
import unittest
import unittest.mock as mock

from . import grad_cam
import numpy as np
import tensorflow.compat.v1 as tf

INPUT_HEIGHT_WIDTH = 5  # width and height of input images in pixels


class GradCamTest(unittest.TestCase):
  """To run: "python -m saliency.tf1.grad_cam_test" from top-level saliency directory."""

  def setUp(self):
    super().setUp()
    self.graph = tf.Graph()
    with self.graph.as_default():
      # Input placeholder
      self.images = tf.placeholder(
          tf.float32, shape=(1, INPUT_HEIGHT_WIDTH, INPUT_HEIGHT_WIDTH, 1))

      # Horizontal line detector filter
      horiz_detector = np.array([[-1, -1, -1],
                                 [2, 2, 2],
                                 [-1, -1, -1]])
      conv1 = tf.keras.layers.Conv2D(
          filters=1,
          kernel_size=3,
          kernel_initializer=tf.constant_initializer(horiz_detector),
          padding='same',
          name='Conv',
          input_shape=self.images.shape)(self.images)

      # Compute logits and do prediction with pre-defined weights
      flat = tf.reshape(conv1, [-1, INPUT_HEIGHT_WIDTH*INPUT_HEIGHT_WIDTH])
      sum_weights = tf.constant_initializer(np.ones(flat.shape))
      logits = tf.keras.layers.Dense(
          units=2, kernel_initializer=sum_weights, name='Logits')(flat)
      y = logits[0][0]
      
      self.sess = tf.Session()
      init = tf.global_variables_initializer()
      self.sess.run(init)
      self.sess_spy = mock.MagicMock(wraps=self.sess)

      # Set up GradCam object
      self.conv_layer = self.graph.get_tensor_by_name('Conv/BiasAdd:0')
      self.grad_cam_instance = grad_cam.GradCam(self.graph,
                                                self.sess_spy,
                                                y=y,
                                                x=self.images,
                                                conv_layer=self.conv_layer)

  def testGradCamGetMask(self):
    """Tests the GradCAM method using a simple network.

    Simple test case where the network contains one convolutional layer that
    acts as a horizontal line detector and the input image is a 5x5 matrix with
    a centered 3x3 grid of 1s and 0s elsewhere.

    The computed GradCAM mask should detect the pixels of highest importance to
    be along the two horizontal lines in the image (exact expected values stored
    in ref_mask).
    """

    # Generate test input (centered matrix of 1s surrounded by 0s)
    # and generate corresponding GradCAM mask
    img = np.zeros([INPUT_HEIGHT_WIDTH, INPUT_HEIGHT_WIDTH])
    img[1:-1, 1:-1] = 1
    img = img.reshape([INPUT_HEIGHT_WIDTH, INPUT_HEIGHT_WIDTH, 1])
    mask = self.grad_cam_instance.GetMask(
        img,
        feed_dict=None,
        should_resize=True,
        three_dims=False)

    # Compare generated mask to expected result
    ref_mask = np.array([[0., 0., 0., 0., 0.],
                         [0.33, 0.67, 1., 0.67, 0.33],
                         [0., 0., 0., 0., 0.],
                         [0.33, 0.67, 1., 0.67, 0.33],
                         [0., 0., 0., 0., 0.]])
    self.assertTrue(np.allclose(mask, ref_mask, atol=0.01),
                    'Generated mask did not match reference mask.')

  def testGradCamGetMaskArgs(self):
    """Tests that sess.run receives the correct inputs."""
    img = np.ones([INPUT_HEIGHT_WIDTH, INPUT_HEIGHT_WIDTH])
    img = img.reshape([INPUT_HEIGHT_WIDTH, INPUT_HEIGHT_WIDTH, 1])
    feed_dict = {'foo': 'bar'}
    self.sess_spy.run.return_value = [[img], [img]]

    self.grad_cam_instance.GetMask(
        img, feed_dict=feed_dict, should_resize=True, three_dims=False)
    actual_feed_dict = self.sess_spy.run.call_args[1]['feed_dict']

    self.assertEqual(actual_feed_dict['foo'], feed_dict['foo'])


if __name__ == '__main__':
  unittest.main()

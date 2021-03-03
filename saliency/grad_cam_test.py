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
from .grad_cam import GradCam
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.python.platform import googletest


class GradCamTest(googletest.TestCase):
  """
  To run:
  "python -m saliency.grad_cam_test" from the PAIR-code/saliency directory.
  """

  def testGradCamGetMask(self):
    """
    Simple test case where the network contains one convolutional layer that
    acts as a horizontal line detector and the input image is a 5x5 matrix with
    a centered 3x3 grid of 1s and 0s elsewhere.

    The computed GradCAM mask should detect the pixels of highest importance to
    be along the two horizontal lines in the image (exact expected values stored
    in ref_mask).
    """
    with tf.Graph().as_default() as graph:
      # Input placeholder
      num_pix = 5 # width and height of input images in pixels
      images = tf.placeholder(tf.float32, shape=(1, num_pix, num_pix, 1))

      # Horizontal line detector filter
      horiz_detector = np.array([[-1,-1,-1],
                                 [ 2, 2, 2],
                                 [-1,-1,-1]])
      conv1 = tf.layers.conv2d(
          inputs = images,
          filters = 1,
          kernel_size = 3,
          kernel_initializer = tf.constant_initializer(horiz_detector),
          padding = "same",
          name = "Conv")

      # Compute logits and do prediction with pre-defined weights
      flat = tf.reshape(conv1,[-1,num_pix*num_pix])
      sum_weights = tf.constant_initializer(np.ones(flat.shape))
      logits = tf.layers.dense(inputs = flat, units = 2,
                               kernel_initializer = sum_weights,
                               name = "Logits")
      predictions = {"classes": tf.argmax(input=logits, axis=1),
                     "probs": tf.nn.softmax(logits, name="softmax")}

      with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        # Set up GradCam object
        logits = graph.get_tensor_by_name("Logits/BiasAdd:0")
        neuron_selector = tf.placeholder(tf.int32)
        y = logits[0][neuron_selector]
        conv_layer = graph.get_tensor_by_name("Conv/BiasAdd:0")

        grad_cam = GradCam(graph, sess, y, images, conv_layer)

        # Generate test input (centered matrix of 1s surrounded by 0s)
        # and generate corresponding GradCAM mask
        img = np.zeros([num_pix,num_pix])
        img[1:-1,1:-1] = 1
        img = img.reshape([num_pix,num_pix,1])
        mask = grad_cam.GetMask(img,
                                feed_dict={neuron_selector: 0},
                                should_resize = True,
                                three_dims = False)

        # Compare generated mask to expected result
        ref_mask = np.array([[0.  , 0.  , 0.  , 0.  , 0.  ],
                             [0.33, 0.67, 1.  , 0.67, 0.33],
                             [0.  , 0.  , 0.  , 0.  , 0.  ],
                             [0.33, 0.67, 1.  , 0.67, 0.33],
                             [0.  , 0.  , 0.  , 0.  , 0.  ]])
        self.assertTrue(np.allclose(mask, ref_mask, atol=0.01),
                        "Generated mask did not match reference mask.")

if __name__ == '__main__':
  googletest.main()

# Copyright 2020 Google Inc. All Rights Reserved.
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

"""Tests completeness axiom for blur_ig.
"""
from . import blur_ig
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.python.platform import googletest


class BlurIgTest(googletest.TestCase):
  """
  To run:
  "python -m saliency.blur_ig_test" from the PAIR-code/saliency directory.
  """

  def testBlurIGGetMask(self):
    max_sigma = 10
    with tf.Graph().as_default() as graph:
      x = tf.placeholder(shape=[None, 5, 5, 1], dtype=tf.float32)
      # Define function to just look at center pixel.
      y = x[:, 2, 2, 0] * 1.0
      with tf.Session() as sess:
        # All black except 1 white pixel at the center.
        x_input_val = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                               ], dtype=np.float)
        x_input_val = x_input_val.reshape((5, 5, 1))
        # Calculate the value of `y` at the input.
        y_input_val = sess.run(y, feed_dict={x: [x_input_val]})

        # Baseline is the fully blurred version of the input.
        x_baseline_val = blur_ig.gaussian_blur(x_input_val, sigma=max_sigma)
        y_baseline_val = sess.run(y, feed_dict={x: [x_baseline_val]})

        # Test if completeness axiom is satisfied.
        # The expected BlurIG value is equal to the difference between
        # the `y` value at the input and the `y` value at the baseline.
        expected_val = y_input_val[0] - y_baseline_val[0]

        # Calculate the Blur IG attribution of the input.
        blur_ig_instance = blur_ig.BlurIG(graph, sess, y[0], x)
        mask = blur_ig_instance.GetMask(
            x_input_val, feed_dict={}, max_sigma=max_sigma, steps=200)
        # Verify the result for completeness..
        # Expected (for max_sigma=10): 0.9984083
        # mask.sum (for max_sigma=10): 0.99832882...
        self.assertAlmostEqual(expected_val, mask.sum(), places=3)


if __name__ == '__main__':
  googletest.main()

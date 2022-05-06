# Copyright 2022 Google Inc. All Rights Reserved.
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

""" Tests for pic.py.

To run: "python -m saliency.metrics.pic_test" from the top-level directory.
"""

import numpy as np
import os
import unittest

from . import pic
from pathlib import Path
from PIL import Image


class PicTest(unittest.TestCase):

  def setUp(self):
    self.dir_path = os.path.dirname(os.path.realpath(__file__))
    self.root_path = Path(self.dir_path).parent.parent.absolute()
    with Image.open(os.path.join(self.root_path, 'doberman.png')) as pil_img:
      self.img = np.array(pil_img)

  def _test_predict_func(self, input_images_batch):
    # For the test purpose, use entropy as the prediction, so images
    # with higher entropy produce higher model predictions. It works only,
    # when the input has only one image in the batch.
    entropy = pic.estimate_image_entropy(input_images_batch[0])
    return [entropy]

  def test_create_blurred_image_full_mask(self):
    """Tests when the mask has only none-zero values.

    The original image should be returned.
    """
    mask = np.full_like(self.img, fill_value=True)[:, :, 0].astype(bool)
    blurred_img = pic.create_blurred_image(full_img=self.img, pixel_mask=mask)
    self.assertTrue(np.all(self.img == blurred_img))

  def test_create_blurred_image_empty_mask(self):
    """Tests when the mask has only zero values.

    The mean color image should be returned.
    """
    mask = np.zeros_like(self.img)[:, :, 0]
    blurred_img = pic.create_blurred_image(full_img=self.img, pixel_mask=mask)
    self.assertIsNotNone(blurred_img)

  def test_entropy_estimate(self):
    """Tests when the mask has only zero values.

    The mean color image should be returned.
    """
    previous_entropy = 0
    for fraction in [0, 0.01, 0.02, 0.1, 1.0]:
      mask = pic.generate_random_mask(image_shape=self.img.shape,
                                      fraction=fraction)
      blurred_img = pic.create_blurred_image(full_img=self.img, pixel_mask=mask)
      entropy = pic.estimate_image_entropy(blurred_img)
      self.assertGreater(entropy, previous_entropy)
      previous_entropy = entropy

  def test_generate_random_mask(self):
    mask = pic.generate_random_mask(image_shape=(100, 100, 3), fraction=0.2)
    self.assertEqual(np.count_nonzero(mask), 2000)

  def test_pic_curve(self):
    num_data_points = 2001
    mask = pic.generate_random_mask(image_shape=self.img.shape, fraction=0.01)
    saliency_map = np.random.random(size=(self.img.shape[0], self.img.shape[1]))
    saliency_thresholds = [0.01, 0.02, 0.03, 0.05, 0.12]
    metric_result = pic.compute_pic_metric(img=self.img,
                                           saliency_map=saliency_map,
                                           random_mask=mask,
                                           pred_func=self._test_predict_func,
                                           saliency_thresholds=saliency_thresholds,
                                           num_data_points=num_data_points)
    self.assertEqual(len(metric_result.curve_x), num_data_points)
    self.assertEqual(len(metric_result.curve_y), num_data_points)
    self.assertGreater(metric_result.auc, 0.0)
    self.assertLess(metric_result.auc, 1.0)
    self.assertEqual(len(metric_result.predictions),
                     len(saliency_thresholds) + 2)
    self.assertEqual(len(metric_result.blurred_images),
                     len(saliency_thresholds) + 2)

  def test_aggregate_individual_image_results(self):
    num_data_points = 501
    individual_results = []
    for _ in range(3):
      mask = pic.generate_random_mask(image_shape=self.img.shape, fraction=0.01)
      saliency_map = np.random.random(
          size=(self.img.shape[0], self.img.shape[1]))
      saliency_thresholds = [0.01, 0.02, 0.03, 0.05, 0.12]
      metric_result = pic.compute_pic_metric(img=self.img,
                                             saliency_map=saliency_map,
                                             random_mask=mask,
                                             pred_func=self._test_predict_func,
                                             saliency_thresholds=saliency_thresholds,
                                             num_data_points=num_data_points)
      individual_results.append(metric_result)

    r = pic.aggregate_individual_pic_results(
        individual_results, method='mean')
    self.assertEqual(len(r.curve_x), num_data_points)
    self.assertEqual(len(r.curve_y), num_data_points)
    self.assertGreater(r.auc, 0.0)
    self.assertLess(r.auc, 1.0)


if __name__ == '__main__':
  unittest.main()

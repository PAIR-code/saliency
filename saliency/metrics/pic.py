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

""" Implementation of Performance Information Curve metrics.

This module implements the saliency evaluation metrics as described in
"XRAI: Better Attributions Through Regions" (1). Use the compute_pic_metric(...)
method for computing both Softmax Information Curve (SIC) and Accuracy
Information Curve (AIC).

Here are the typical steps to compute the aggregated PIC curve for saliency
methods:
1) Call generate_random_mask(...) to generate a random pixel mask that serves
   as the seed for generating initial fully blurred image. To avoid using the
   same random mask for all images, generate a new mask for every new image.
   For fair comparison of multiple saliency methods, use the same random mask
   on a given image. If exact reproducibility is required, save the random mask
   on disk along with the results and reuse the masks on subsequent runs.
   The result of generate_random_mask(...) should be passed as the `random_mask`
   argument to compute_pic_metric(...).
2) Call compute_pic_metric(...) for every image and store the results in a list.
   Use separate lists for different saliency methods. Be aware, that the
   method can raise ComputePicMetricError. If that happens, skip the image.
   At the end of this step, you should have multiple lists with results: one
   list per saliency method.
3) Aggregate the results obtained in step 2 by calling
   aggregate_individual_pic_results(...). One call should be made per saliency
   method (i.e. per list obtained in step 2).

See the complementary pic_metrics.ipynb Jupyter notebook for an example of
usage.

(1) https://arxiv.org/abs/1906.02825
"""

import io
import numpy as np
import os

from PIL import Image
from scipy import interpolate
from typing import Callable, List, NamedTuple, Optional, Sequence, Tuple


def create_blurred_image(full_img: np.ndarray, pixel_mask: np.ndarray,
    method: str = 'linear') -> np.ndarray:
  """ Creates a blurred (interpolated) image.

  Args:
    full_img: an original input image that should be used as the source for
      interpolation. The image should be represented by a numpy array with
      dimensions [H, W, C] or [H, W].
    pixel_mask: a binary mask, where 'True' values represent pixels that should
      be retrieved from the original image as the source for the interpolation
      and 'False' values represent pixels, which values should be found. The
      method always sets the corner pixels of the mask to True. The mask
      dimensions should be [H, W].
    method: the method to use for the interpolation. The 'linear' method is
      recommended. The alternative value is 'nearest'.

    Returns:
      A numpy array that encodes the blurred image with exactly the same
      dimensions and type as `full_img`.
  """
  data_type = full_img.dtype
  has_color_channel = full_img.ndim > 2
  if not has_color_channel:
    full_img = np.expand_dims(full_img, axis=2)
  channels = full_img.shape[2]

  # Always include corners.
  pixel_mask = pixel_mask.copy()
  height = pixel_mask.shape[0]
  width = pixel_mask.shape[1]
  pixel_mask[
    [0, 0, height - 1, height - 1], [0, width - 1, 0, width - 1]] = True

  mean_color = np.mean(full_img, axis=(0, 1))

  # If the mask consists of all pixels set to True then return the original
  # image.
  if np.all(pixel_mask):
    return full_img

  blurred_img = full_img * np.expand_dims(pixel_mask, axis=2).astype(
      np.float32)

  # Interpolate the unmasked values of the image pixels.
  for channel in range(channels):
    data_points = np.argwhere(pixel_mask > 0)
    data_values = full_img[:, :, channel][tuple(data_points.T)]
    unknown_points = np.argwhere(pixel_mask == 0)
    interpolated_values = interpolate.griddata(np.array(data_points),
                                               np.array(data_values),
                                               np.array(unknown_points),
                                               method=method,
                                               fill_value=mean_color[channel])
    blurred_img[:, :, channel][tuple(unknown_points.T)] = interpolated_values

  if not has_color_channel:
    blurred_img = blurred_img[:, :, 0]

  if issubclass(data_type.type, np.integer):
    blurred_img = np.round(blurred_img)

  return blurred_img.astype(data_type)


def generate_random_mask(image_height: int, image_width: int,
    fraction=0.01) -> np.ndarray:
  """Generates a random pixel mask.

    The result of this method can be used as the initial seed to generate
    a 'fully' blurred image with the help of the create_blurred_image(...)
    method.

    Args:
      image_height: the image height for which the mask should be generated.
      image_width: the image width for which the mask should be generated.
      fraction: the fraction of the mask pixels that should be set to true.
        The valid value range is [0.0, 1.0]. Set the value to 0.0 if no
        information from the original image should be used in the blurred image.
    Returns:
      The binary mask with the `fraction` of elements set to True.
  """
  mask = np.zeros(shape=[image_height, image_width], dtype=bool)
  size = mask.size
  indices = np.random.choice(size, replace=False, size=int(size * fraction))
  mask[np.unravel_index(indices, mask.shape)] = True
  return mask


def estimate_image_entropy(image: np.ndarray) -> float:
  """Estimates the amount of information in a given image.

    Args:
      image: an image, which entropy should be estimated. The dimensions of the
        array should be [H, W, C] or [H, W] of type uint8.
    Returns:
      The estimated amount of information in the image.
  """
  buffer = io.BytesIO()
  pil_image = Image.fromarray(image)
  pil_image.save(buffer, format='webp', lossless=True, quality=100)
  buffer.seek(0, os.SEEK_END)
  length = buffer.tell()
  buffer.close()
  return length


class ComputePicMetricError(Exception):
  """An error that can be raised by the compute_pic_metric(...) method.

  See the method description for more information.
  """
  pass


class PicMetricResult(NamedTuple):
  """Holds results of compute_pic_metric(...) method."""
  # x-axis coordinates of PIC curve data points.
  curve_x: Sequence[float]
  # y-axis coordinates of PIC curve data points.
  curve_y: Sequence[float]
  # A sequence of intermediate blurred images used for PIC computation with
  # the fully blurred image in front and the original image at the end.
  blurred_images: Sequence[np.ndarray]
  # Model predictions for images in the `blurred_images` sequence.
  predictions: Sequence[float]
  # Saliency thresholds that were used to generate corresponding
  # `blurred_images`.
  thresholds: Sequence[float]
  # Area under the curve.
  auc: float


def compute_pic_metric(
    img: np.ndarray,
    saliency_map: np.ndarray,
    random_mask: np.ndarray,
    pred_func: Callable[[np.ndarray], Sequence[float]],
    saliency_thresholds: Sequence[float],
    min_pred_value: float = 0.8,
    keep_monotonous: bool = True,
    num_data_points: int = 1000
) -> PicMetricResult:
  """Computes Performance Information Curve for a single image.

    The method can be used to compute either Softmax Information Curve (SIC) or
    Accuracy Information Curve (AIC). The method computes the curve for a single
    image and saliency map. This method should be called repeatedly on different
    images and saliency maps.

    Args:
      img: an original image on which the curve should be computed. The numpy
        array should have dimensions [H, W, C] for a color image or [H, W]
        for a grayscale image. The array should be of type uint8.
      saliency_map: the saliency map for which the metric should be calculated.
        Usually, the saliency map should be calculated with respect to the
        same class label as the class label for which `pred_func` returns the
        prediction. However, the class labels may be different if you want
        to see how saliency for one class explains prediction of other class.
        Pixels with higher values are considered to be of higher importance.
        It is the responsibility of the caller to decide on the order of pixel
        importance, e.g. if the absolute values should be used instead of the
        signed ones, the caller should apply 'abs' function before calling this
        method. The shape of `saliency_map` is [H, W].
      random_mask: a random mask to use in order to create the initial
        completely blurred image.
      pred_func: a function that the method should call in order to obtain
        the prediction of a model on a batch of images. The method should accept
        a numpy array of type uint8 with dimensions [B, H, W, C] for color
        images or [B, H, W] for grayscale images and return the predictions
        as a list of float values. For multiclass classification, it is
        up to the implementer of the function to return the prediction for
        the desired class. For the SIC metric, the function should return the
        softmax of the corresponding class. For the AIC metric, the function
        should return 1.0 if the classification was correct and 0.0 otherwise.
        Note, that the images are passed as an uint8 array. If the underlying
        model requires a different format, it is the responsibility of the
        function implementer to do the corresponding conversion inside
        `pred_func`. See the accompanying Jupyter notebook for an example.
      saliency_thresholds: the thresholds represent the fractions of the most
        important saliency pixels that will be used to reconstruct
        intermediate blurred images and pass them to the model for evaluation.
        The value of this argument should be the list of thresholds in
        ascending order. Example value: [0.005, 0.01, 0.02, 0.03, 0.04, 0.05,
        0.07, 0.10, 0.13, 0.21, 0.34, 0.5, 0.75].
      min_pred_value: used for filtering images. If the model prediction on the
        original image is lower than the value of this argument, the method
        raises ComputePicMetricError to indicate that the image should be
        skipped. This is done to filter out images that produce low prediction
        confidence.
      keep_monotonous: whether to keep the curve monotonically increasing.
        The value of this argument was set to 'True' in the original paper but
        setting it to 'False' is a viable alternative.
      num_data_points: the number of PIC curve data points to return. The number
        excludes point 1.0 on the x-axis that is always appended to the end.
        E.g., value 1000 results in 1001 points evently distributed on the
        x-axis from 0.0 to 1.0 with 0.001 increment.

    Returns:
      The PIC curve data points and extra auxiliary information. See
      `PicMetricResult` for more information.

    Raises:
      ComputePicMetricError:
        The method raises the error in two cases. That happens in two cases:
        1. If the model prediction on the original image is not higher than the
           model prediction on the completely blurred image.
        2. If the entropy of the original image is not higher than the entropy
           of the completely blurred image.
        3. If the model prediction on the original image is lower than
           `min_pred_value`.
        If the error is raised, skip the image.
  """
  if img.dtype.type != np.uint8:
    raise ValueError('The `img` array that holds the input image should be of'
                     ' type uint8. The actual type is {}.'.format(img.dtype))
  blurred_images = []
  predictions = []

  # This list will contain mapping of image entropy for a given saliency
  # threshold to model prediction.
  entropy_pred_tuples = []

  # Estimate entropy of the original image.
  original_img_entropy = estimate_image_entropy(img)

  # Estimate entropy of the completely blurred image.
  fully_blurred_img = create_blurred_image(full_img=img, pixel_mask=random_mask)
  fully_blurred_img_entropy = estimate_image_entropy(fully_blurred_img)

  # Compute model prediction for the original image.
  original_img_pred = pred_func(img[np.newaxis, ...])[0]

  if original_img_pred < min_pred_value:
    message = ('The model prediction score on the original image is lower than'
               ' `min_pred_value`. Skip this image or decrease the'
               ' value of `min_pred_value` argument. min_pred_value'
               ' = {}, the image prediction'
               ' = {}.'.format(min_pred_value, original_img_pred))
    raise ComputePicMetricError(message)

  # Compute model prediction for the completely blurred image.
  fully_blurred_img_pred = pred_func(fully_blurred_img[np.newaxis, ...])[0]

  blurred_images.append(fully_blurred_img)
  predictions.append(fully_blurred_img_pred)

  # If the entropy of the completely blurred image is higher or equal to the
  # entropy of the original image then the metric cannot be used for this
  # image. Don't include this image in the aggregated result.
  if fully_blurred_img_entropy >= original_img_entropy:
    message = (
        'The entropy in the completely blurred image is not lower than'
        ' the entropy in the original image. Catch the error and exclude this'
        ' image from evaluation. Blurred entropy: {}, original'
        ' entropy {}'.format(fully_blurred_img_entropy, original_img_entropy))
    raise ComputePicMetricError(message)

  # If the score of the model on completely blurred image is higher or equal to
  # the score of the model on the original image then the metric cannot be used
  # for this image. Don't include this image in the aggregated result.
  if fully_blurred_img_pred >= original_img_pred:
    message = (
        'The model prediction score on the completely blurred image is not'
        ' lower than the score on the original image. Catch the error and'
        ' exclude this image from the evaluation. Blurred score: {}, original'
        ' score {}'.format(fully_blurred_img_pred, original_img_pred))
    raise ComputePicMetricError(message)

  # Iterate through saliency thresholds and compute prediction of the model
  # for the corresponding blurred images with the saliency pixels revealed.
  max_normalized_pred = 0.0
  for threshold in saliency_thresholds:
    quantile = np.quantile(saliency_map, 1 - threshold)
    pixel_mask = saliency_map >= quantile
    pixel_mask = np.logical_or(pixel_mask, random_mask)
    blurred_image = create_blurred_image(full_img=img, pixel_mask=pixel_mask)
    entropy = estimate_image_entropy(blurred_image)
    pred = pred_func(blurred_image[np.newaxis, ...])[0]
    # Normalize the values, so they lie in [0, 1] interval.
    normalized_entropy = (entropy - fully_blurred_img_entropy) / (
        original_img_entropy - fully_blurred_img_entropy)
    normalized_entropy = np.clip(normalized_entropy, 0.0, 1.0)
    normalized_pred = (pred - fully_blurred_img_pred) / (
        original_img_pred - fully_blurred_img_pred)
    normalized_pred = np.clip(normalized_pred, 0.0, 1.0)
    max_normalized_pred = max(max_normalized_pred, normalized_pred)

    # Make normalized_pred only grow if keep_monotonous is true.
    if keep_monotonous:
      entropy_pred_tuples.append((normalized_entropy, max_normalized_pred))
    else:
      entropy_pred_tuples.append((normalized_entropy, normalized_pred))

    blurred_images.append(blurred_image)
    predictions.append(pred)

  # Interpolate the PIC curve.
  entropy_pred_tuples.append((0.0, 0.0))
  entropy_pred_tuples.append((1.0, 1.0))

  entropy_data, pred_data = zip(*entropy_pred_tuples)
  interp_func = interpolate.interp1d(x=entropy_data, y=pred_data)

  curve_x = np.linspace(start=0.0, stop=1.0, num=num_data_points,
                        endpoint=False)
  curve_y = np.asarray([interp_func(x) for x in curve_x])

  curve_x = np.append(curve_x, 1.0)
  curve_y = np.append(curve_y, 1.0)

  auc = np.trapz(curve_y, curve_x)

  blurred_images.append(img)
  predictions.append(original_img_pred)

  thresholds = [0.0] + list(saliency_thresholds) + [1.0]

  return PicMetricResult(curve_x=curve_x, curve_y=curve_y,
                         blurred_images=blurred_images,
                         predictions=predictions, thresholds=thresholds,
                         auc=auc)


class AggregateMetricResult(NamedTuple):
  """Holds results of aggregate_individual_pic_results(...) method."""
  # x-axis coordinates of aggregated PIC curve data points.
  curve_x: Sequence[float]
  # y-axis coordinates of aggregated PIC curve data points.
  curve_y: Sequence[float]
  # Area under the curve.
  auc: float


def aggregate_individual_pic_results(
    compute_pic_metrics_results: List[PicMetricResult],
    method: str = 'median') -> AggregateMetricResult:
  """Aggregates PIC metrics of individual images to produce the aggregate curve.

    The method should be called after calling the compute_pic_metric(...) method
    on multiple images for a given single saliency method.

    Args:
      compute_pic_metrics_results: a list of PicMetricResult instances that are
        obtained by calling compute_pic_metric(...) on multiple images.
      method: method to use for the aggregation. The valid values are 'mean' and
        'median'.
    Returns:
      AggregateMetricResult - a tuple with x, y coordinates of the curve along
        with the AUC value.


  """
  if not compute_pic_metrics_results:
    raise ValueError('The list of results should have at least one element.')

  curve_ys = [r.curve_y for r in compute_pic_metrics_results]
  curve_ys = np.asarray(curve_ys)

  # Validate that x-axis points for all individual results are the same.
  curve_xs = [r.curve_x for r in compute_pic_metrics_results]
  curve_xs = np.asarray(curve_xs)
  _, counts = np.unique(curve_xs, axis=1, return_counts=True)
  if not np.all(counts == 1):
    raise ValueError('Individual results have different x-axis data points.')

  if method == 'mean':
    aggr_curve_y = np.mean(curve_ys, axis=0)
  elif method == 'median':
    aggr_curve_y = np.median(curve_ys, axis=0)
  else:
    raise ValueError('Unknown method {}.'.format(method))

  auc = np.trapz(aggr_curve_y, curve_xs[0])

  return AggregateMetricResult(curve_x=curve_xs[0], curve_y=aggr_curve_y,
                               auc=auc)

"""Implementation of segment integrated gradients code.

This code is based on the structure of third_party/py/saliency.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging

import numpy as np
import saliency
from skimage import segmentation
from skimage.morphology import dilation
from skimage.morphology import disk
from skimage.transform import resize


def _normalize_image(im, value_range, resize_shape=None):
  im_max = np.max(im)
  im_min = np.min(im)
  im = (im - im_min) / (im_max - im_min)
  im -= 0.5
  im *= value_range[1] - value_range[0]
  im += np.mean(value_range)
  if resize_shape is not None:
    im = resize(im,
                resize_shape,
                order=3,
                mode='constant',
                preserve_range=True,
                anti_aliasing=True)
  return im


def _get_segments_felsenschwab(im,
                               resize_image=True,
                               scale_range=[-1.0, 1.0],
                               dilation_rad=5):
  """Get an image and return segments based on felsenschwab algorithm
  TODO (tolgab) This resize is unnecessary with more intelligent param range
  selection
  If resize_image is True, images are resized to 224,224 for segmentation
    output segments are always the same size as a the input image. This is
    for consistency w.r.t. segmentation parameter range
  TODO (tolgab) Set this to default float range of 0.0 - 1.0 and tune parameters
  for that
  scale_range is the range of image values to use for segmentation algorithm.
  Segmentation algorithm is sensitive to the input image values, therefore we
  need to be consistent with the range for all images.
  dilation_rad sets how much each segment is dilated to include edges, larger
  values cause more blobby segments, smaller values get sharper areas.
  """
  SCALE_VALUES = [50, 100, 150, 250, 500, 1200]
  SIGMA_VALUES = [0.8]
  # Normalize image value range and size
  original_shape = im.shape[:2]
  if resize_image:
    im = _normalize_image(im, scale_range, (224, 224))
  else:
    im = _normalize_image(im, scale_range)
  segs = []
  for scale in SCALE_VALUES:
    for sigma in SIGMA_VALUES:
      seg = segmentation.felzenszwalb(im, scale=scale, sigma=sigma, min_size=20)
      if resize_image:
        seg = resize(seg,
                     original_shape,
                     order=0,
                     preserve_range=True,
                     mode='constant',
                     anti_aliasing=False).astype(np.uint8)
      segs.append(seg)
  masks = _unpack_segs_to_masks(segs)
  if dilation_rad:
    selem = disk(dilation_rad)
    masks = [dilation(mask, selem=selem) for mask in masks]
  return masks


def _accumulate_attr_max(attr, axis=-1):
  return attr.max(axis=axis)


def _gain_density(mask1, attr, mask2=None):
  # Compute the attr density over mask1. If mask2 is specified, compute density
  # for mask1 \ mask2
  if mask2 is None:
    added_mask = mask1
  else:
    added_mask = np.logical_and(mask1, np.logical_not(mask2))
  return attr[added_mask].mean()


def _get_iou(mask1, mask2):
  return (np.sum(np.logical_and(mask1, mask2)) /
          np.sum(np.logical_or(mask1, mask2)))


def _get_diff_cnt(add_mask, base_mask):
  return np.sum(np.logical_and(add_mask, np.logical_not(base_mask)))


def _unpack_segs_to_masks(segs):
  masks = []
  for seg in segs:
    for l in xrange(seg.min(), seg.max() + 1):
      masks.append(seg == l)
  return masks


class XRAIParameters(object):

  def __init__(self,
               steps=100,
               area_threshold=1.0,
               return_baseline_predictions=False,
               return_ig_attributions=False,
               return_xrai_segments=False,
               flatten_xrai_segments=True,
               algorithm='full',
               verbosity=0):
    # TODO(tolgab) add return_ig_for_every_step functionality

    # Number of steps to use for calculating the Integrated Gradients
    # attribution. The higher the number of steps the higher is the precision
    # but lower the performance. (see also XRAIOutput.error).
    self.steps = steps
    # The fraction of the image area that XRAI should calculate the segments
    # for. All segments that exceed that threshold will be merged into a single
    # segment. The parameter is used to accelerate the XRAI computation if the
    # caller is only interested in the top fraction of segments, e.g. 20%. The
    # value should be in the [0.0, 1.0] range, where 1.0 means that all segments
    # should be returned (slowest). Fast algorithm ignores this setting.
    self.area_threshold = area_threshold
    # If set to True returns predictions for the baselines as float32 [B] array,
    # where B is the number of baselines. (see XraiOutput.baseline_predictions).
    self.return_baseline_predictions = return_baseline_predictions
    # If set to True, the XRAI output returns Integrated Gradients attributions
    # for every baseline. (see XraiOutput.ig_attribution)
    self.return_ig_attributions = return_ig_attributions
    # If set to True the XRAI output returns XRAI segments in the order of their
    # importance. This parameter works in conjunction with the
    # flatten_xrai_sements parameter. (see also XraiOutput.segments)
    self.return_xrai_segments = return_xrai_segments
    # If set to True, the XRAI segments are returned as an integer array with
    # the same dimensions as the input (excluding color channels). The elements
    # of the array are set to values from the [1,N] range, where 1 is the most
    # important segment and N is the least important segment. If
    # flatten_xrai_sements is set to False, the segments are returned as a
    # boolean array, where the first dimension has size N. The [0, ...] mask is
    # the most important and the [N-1, ...] mask is the least important. This
    # parameter has an effect only if return_xrai_segments is set to True.
    self.flatten_xrai_segments = flatten_xrai_segments
    # Specifies a flavor of the XRAI algorithm. full - executes slower but more
    # precise XRAI algorithm. fast - executes faster but less precise XRAI
    # algorithm.
    self.algorithm = algorithm
    # Specifies the level of verbosity. 0 - silent.
    self.verbosity = verbosity


class SaliencyOutput(object):

  def __init__(self, attribution_mask):
    # The saliency mask of individual input features. For an [NxMx3] image, the
    # returned attribution is [N,M,1] float32 array. Where NxM are the
    # dimensions of the image.
    self.attribution_mask = attribution_mask


class XRAIOutput(SaliencyOutput):

  def __init__(self, attribution_mask):
    super(XRAIOutput, self).__init__(attribution_mask)
    # Baselines that were used for IG calculation. The shape is [B,N,M], where B
    # is the number of baselines, NxM are the image dimensions.
    self.baselines = None
    # The average error of the IG attributions as a percentage. The error can be
    # decreased by increasing the number of steps (see XraiParameters.steps).
    self.error = None
    # Predictions for the baselines that were used for the calculation of IG
    # attributions. The value is set only when
    # XraiParameters.return_baseline_predictions is set to True.
    self.baseline_predictions = None
    # IG attributions for individual baselines. The value is set only when
    # XraiParameters.ig_attributions is set to True. For the dimensions of the
    # output see XraiParameters.return_ig_for_every _step.
    self.ig_attribution = None
    # The result of the XRAI segmentation. The value is set only when
    # XraiParameters.return_xrai_segments is set to True. For the dimensions of
    # the output see XraiParameters.flatten_xrai_segments.
    self.segments = None


class XRAI(saliency.GradientSaliency):

  def __init__(self, graph, session, y, images):
    # Initialize integrated gradients
    self._integrated_gradients = saliency.IntegratedGradients(
        graph, session, y, images)

  def _get_integrated_gradients(self, im, feed_dict, baselines, steps):
    """ Takes mean of attributions from all baselines
    """
    grads = []
    for baseline in baselines:
      grads.append(
          self._integrated_gradients.GetMask(im,
                                             feed_dict=feed_dict,
                                             x_baseline=baseline,
                                             x_steps=steps))
    return grads

  def _make_baselines(self, x_value, x_baselines):
    # If baseline is not provided default to im min and max values
    if x_baselines is None:
      x_baselines = []
      x_baselines.append(np.min(x_value) * np.ones_like(x_value))
      x_baselines.append(np.max(x_value) * np.ones_like(x_value))
    else:
      for baseline in x_baselines:
        if baseline.shape != x_value.shape:
          raise ValueError(
              "Baseline size {} does not match input size {}".format(
                  baseline.shape, x_value.shape))
    return x_baselines

  def _predict(self, x):
    raise NotImplementedError

  def GetMask(self,
              x_value,
              feed_dict={},
              baselines=None,
              segments=None,
              extra_parameters=None):
    """ Output a np.ndarray heatmap of XRAI attributions with input shape.
    """
    results = self.GetMaskWithDetails(x_value,
                                      feed_dict=feed_dict,
                                      baselines=baselines,
                                      segments=segments,
                                      extra_parameters=extra_parameters)
    return results.attribution_mask

  def GetMaskWithDetails(self,
                         x_value,
                         feed_dict={},
                         baselines=None,
                         segments=None,
                         extra_parameters=None):
    """ Parameters:
          x_value - input value, not batched.
          output_selector=None - the index of the output to calculate the
                                 saliency for in the output tensor.
          feed_dict=None - feed dictionary to pass to the TF session.run() call.
          baselines=None - a list of baselines to use for calculating
                           Integrated Gradients attribution. Every baseline in
                           the list should have the same dimensions as the
                           input. If the value is not set then the algorithm
                           will make the best effort to select default
                           baselines.
          segments=None - the list of precalculated image segments that should
                          be passed to XRAI. Each element of the list is an
                          [N,M] integer array, where NxM are the image
                          dimensions. Each element of the list may provide
                          information about multiple segments by encoding them
                          with distinct integer values. If the value is None,
                          a defaut segmentation algorithm will be applied.
          extra_parameters=None - a XraiParameters object that specifies
                                  additional parameters for the XRAI saliency
                                  method.

        Returns:
          a XraiOutput object that contains the output of the XRAI algorithm.

    TODO(tolgab) Add output_selector functionality from XRAI API doc
    """
    if extra_parameters.verbosity > 1:
      logging.info("Computing IG...")
    x_baselines = self._make_baselines(x_value, baselines)

    attrs = self._get_integrated_gradients(x_value,
                                           feed_dict=feed_dict,
                                           baselines=x_baselines,
                                           steps=extra_parameters.steps)
    # Merge attributions from different baselines
    attr = np.mean(attrs, axis=0)
    # Merge attribution channels for XRAI input
    attr = _accumulate_attr_max(attr)

    if extra_parameters.verbosity > 1:
      logging.info("Done with IG. Computing XRAI...")
    if segments is not None:
      segs = segments
    else:
      segs = _get_segments_felsenschwab(x_value)

    if extra_parameters.algorithm == 'full':
      attr_map, attr_data = self._xrai(
          attr=attr,
          segs=segs,
          area_perc_th=extra_parameters.area_threshold,
          gain_fun=_gain_density,
          verbose=extra_parameters.verbosity,
          integer_segments=extra_parameters.flatten_xrai_segments)
    elif extra_parameters.algorithm == 'fast':
      attr_map, attr_data = self._xrai_fast(
          attr=attr,
          segs=segs,
          gain_fun=_gain_density,
          verbose=extra_parameters.verbosity,
          integer_segments=extra_parameters.flatten_xrai_segments)
    else:
      logging.error('Unknown algorithm type: {}'.format(
          extra_parameters.algorithm))
      raise ValueError

    results = XRAIOutput(attr_map)
    results.baselines = x_baselines
    if extra_parameters.return_xrai_segments:
      results.segments = attr_data
    if extra_parameters.return_baseline_predictions:
      baseline_predictions = []
      for baseline in x_baselines:
        baseline_predictions.append(self._predict(baseline))
      results.baseline_predictions = baseline_predictions
    if extra_parameters.return_ig_attributions:
      results.ig_attribution = attr
    return results

  @staticmethod
  def _xrai(attr,
            segs,
            gain_fun=_gain_density,
            area_perc_th=1.0,
            verbose=0,
            min_pixel_diff=20,
            integer_segments=True):
    """We expect attr to be 2D, XRAI shape is equal to attr shape
      Segs are list of binary masks, one per segment (pre-dilated if neeeded)
    """
    output_attr = -np.inf * np.ones(shape=attr.shape, dtype=np.float)

    n_masks = len(segs)
    current_attr_sum = 0.0
    current_area_perc = 0.0
    current_mask = np.zeros(attr.shape, dtype=bool)

    masks_trace = []
    attr_ranks = np.zeros(shape=attr.shape, dtype=np.int)
    remaining_masks = {ind: mask for ind, mask in enumerate(segs)}

    added_masks_cnt = 1
    # While the mask area is less than area_th and remaining_masks is not empty
    while current_area_perc <= area_perc_th:
      best_gain = -np.inf
      best_key = None
      remove_key_queue = []
      for mask_key, mask in remaining_masks.iteritems():
        # If mask overlaps current mask more than max_iou then delete it
        mask_pixel_diff = _get_diff_cnt(mask, current_mask)
        if mask_pixel_diff < min_pixel_diff:
          remove_key_queue.append(mask_key)
          if verbose > 2:
            logging.info("Skipping mask with pixel difference: {:.3g},".format(
                mask_pixel_diff))
          continue
        gain = gain_fun(mask, attr, mask2=current_mask)
        if gain > best_gain:
          best_gain = gain
          best_key = mask_key
      for key in remove_key_queue:
        del remaining_masks[key]
      if len(remaining_masks) == 0:
        break
      added_mask = remaining_masks[best_key]
      mask_diff = np.logical_and(np.logical_not(current_mask), added_mask)
      if not integer_segments:
        masks_trace.append(added_mask)
      else:
        attr_ranks[mask_diff] = added_masks_cnt
      current_mask = np.logical_or(current_mask, added_mask)
      current_attr_sum = np.sum(attr[current_mask])
      current_area_perc = np.mean(current_mask)
      output_attr[mask_diff] = best_gain
      del remaining_masks[best_key]  # delete used key
      if verbose:
        logging.info(
            "{} of {} masks added,"
            "attr_sum: {}, area: {:.3g}/{:.3g}, {} remaining masks".format(
                added_masks_cnt, n_masks, current_attr_sum, current_area_perc,
                area_perc_th, len(remaining_masks)))
      added_masks_cnt += 1

    ig_sum = np.sum(attr)
    uncomputed_mask = output_attr == -np.inf
    assert np.all(uncomputed_mask == (attr_ranks == 0))
    # Assign the uncomputed areas a value such that sum is same as ig
    output_attr[uncomputed_mask] = (ig_sum -
                                    current_attr_sum) / np.sum(uncomputed_mask)
    # Set uncomputed region's rank to max rank + 1
    attr_ranks[uncomputed_mask] = np.max(attr_ranks) + 1
    if integer_segments:
      return output_attr, attr_ranks
    else:
      return output_attr, masks_trace

  @staticmethod
  def _xrai_fast(attr,
                 segs,
                 gain_fun=_gain_density,
                 area_perc_th=1.0,
                 verbose=0,
                 integer_segments=True):
    """We expect attr to be 2D, XRAI shape is equal to attr shape
      Segs are list of binary masks, one per segment (pre-dilated if neeeded)
    """
    output_attr = -np.inf * np.ones(shape=attr.shape, dtype=np.float)

    n_masks = len(segs)
    current_attr_sum = 0.0
    current_area_perc = 0.0
    current_mask = np.zeros(attr.shape, dtype=bool)

    masks_trace = []
    attr_ranks = np.zeros(shape=attr.shape, dtype=np.int)

    # Sort all masks based on gain, ignore overlaps
    attr_sums = [gain_fun(seg_mask, attr) for seg_mask in segs]
    sorted_inds, sorted_sums = zip(
        *sorted(zip(range(len(attr_sums)), attr_sums), key=lambda x: -x[1]))
    sorted_inds = np.array(sorted_inds, dtype=np.int)
    segs = np.array(segs)
    segs = segs[sorted_inds]

    for i, added_mask in enumerate(segs):
      mask_diff = np.logical_and(np.logical_not(current_mask), added_mask)
      if not integer_segments:
        masks_trace.append(added_mask)
      else:
        attr_ranks[mask_diff] = i + 1
      current_mask = np.logical_or(current_mask, added_mask)
      current_attr_sum = np.sum(attr[current_mask])
      current_area_perc = np.mean(current_mask)
      output_attr[mask_diff] = sorted_sums[i]
      if verbose:
        logging.info("{} of {} masks added,"
                     "attr_sum: {}, area: {:.3g}/{:.3g}".format(
                         i, n_masks, current_attr_sum, current_area_perc,
                         area_perc_th))
    if integer_segments:
      return output_attr, attr_ranks
    else:
      return output_attr, masks_trace

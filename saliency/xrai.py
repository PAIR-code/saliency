"""Implementation of XRAI algorithm from the paper:
https://arxiv.org/abs/1906.02825
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging

import numpy as np
from skimage import segmentation
from skimage.morphology import dilation
from skimage.morphology import disk
from skimage.transform import resize

from .base import SaliencyMask
from .integrated_gradients import IntegratedGradients


def _normalize_image(im, value_range, resize_shape=None):
  im_max = np.max(im)
  im_min = np.min(im)
  im = (im - im_min) / (im_max - im_min)
  im = im * (value_range[1] - value_range[0]) + value_range[0]
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
                               scale_range=None,
                               dilation_rad=5):
  """Compute image segments based on felsenschwab algorithm

  Args:
    im: Input image.
    resize_image: If resize_image is True, images are resized to 224,224 for
                  segmentation output segments are always the same size as a
                  the input image. This is for consistency w.r.t. segmentation
                  parameter range. Defaults to True.
    scale_range: Range of image values to use for segmentation algorithm.
                  Segmentation algorithm is sensitive to the input image
                  values, therefore we need to be consistent with the range
                  for all images. Defaults to None.
    dilation_rad: Sets how much each segment is dilated to include edges,
                  larger values cause more blobby segments, smaller values
                  get sharper areas. Defaults to 5.
  Returns:
      masks: List of np.ndarrays size of HxW for im size of HxWxC
  """

  # TODO (tolgab) Set this to default float range of 0.0 - 1.0 and tune
  # parameters for that
  if scale_range is None:
    scale_range = [-1.0, 1.0]
  SCALE_VALUES = [50, 100, 150, 250, 500, 1200]
  SIGMA_VALUES = [0.8]
  # Normalize image value range and size
  original_shape = im.shape[:2]
  # TODO (tolgab) This resize is unnecessary with more intelligent param range
  # selection
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
                     anti_aliasing=False).astype(np.int)
      segs.append(seg)
  masks = _unpack_segs_to_masks(segs)
  if dilation_rad:
    selem = disk(dilation_rad)
    masks = [dilation(mask, selem=selem) for mask in masks]
  return masks


def _attr_aggregation_max(attr, axis=-1):
  return attr.max(axis=axis)


def _gain_density(mask1, attr, mask2=None):
  # Compute the attr density over mask1. If mask2 is specified, compute density
  # for mask1 \ mask2
  if mask2 is None:
    added_mask = mask1
  else:
    added_mask = _get_diff_mask(mask1, mask2)
  return attr[added_mask].mean()


def _get_diff_mask(add_mask, base_mask):
  return np.logical_and(add_mask, np.logical_not(base_mask))


def _get_diff_cnt(add_mask, base_mask):
  return np.sum(_get_diff_mask(add_mask, base_mask))


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


class XRAIOutput(object):

  def __init__(self, attribution_mask):
    # The saliency mask of individual input features. For an [HxWx3] image, the
    # returned attribution is [H,W,1] float32 array. Where HxW are the
    # dimensions of the image.
    self.attribution_mask = attribution_mask
    # Baselines that were used for IG calculation. The shape is [B,H,W,C], where
    # B is the number of baselines, HxWxC are the image dimensions.
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


class XRAI(SaliencyMask):

  def __init__(self, graph, session, y, x):
    # Initialize integrated gradients
    super(XRAI, self).__init__(graph, session, y, x)
    self._integrated_gradients = IntegratedGradients(
        graph, session, y, x)

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
    """ Applies XRAI method on an input image and returns the result saliency
    heatmap.


    Args:
        x_value: input value, not batched.
        feed_dict: feed dictionary to pass to the TF session.run() call.
                   Defaults to {}.
        baselines: a list of baselines to use for calculating
                   Integrated Gradients attribution. Every baseline in
                   the list should have the same dimensions as the
                   input. If the value is not set then the algorithm
                   will make the best effort to select default
                   baselines. Defaults to None.
        segments: the list of precalculated image segments that should
                  be passed to XRAI. Each element of the list is an
                  [N,M] integer array, where NxM are the image
                  dimensions. Each element of the list may provide
                  information about multiple segments by encoding them
                  with distinct integer values. If the value is None,
                  a defaut segmentation algorithm will be applied. Defaults to
                  None.
        extra_parameters: an XRAIParameters object that specifies
                          additional parameters for the XRAI saliency
                          method. Defaults to None.

    Raises:
        ValueError: If algorithm type is unknown (not full or fast)

    Returns:
        np.ndarray: A numpy array that contains the saliency heatmap.


    TODO(tolgab) Add output_selector functionality from XRAI API doc
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
    """Applies XRAI method on an input image and returns the result saliency
    heatmap along with other detailed information.


    Args:
        x_value: input value, not batched.
        feed_dict: feed dictionary to pass to the TF session.run() call.
                   Defaults to {}.
        baselines: a list of baselines to use for calculating
                   Integrated Gradients attribution. Every baseline in
                   the list should have the same dimensions as the
                   input. If the value is not set then the algorithm
                   will make the best effort to select default
                   baselines. Defaults to None.
        segments: the list of precalculated image segments that should
                  be passed to XRAI. Each element of the list is an
                  [N,M] integer array, where NxM are the image
                  dimensions. Each element of the list may provide
                  information about multiple segments by encoding them
                  with distinct integer values. If the value is None,
                  a defaut segmentation algorithm will be applied. Defaults to
                  None.
        extra_parameters: an XRAIParameters object that specifies
                          additional parameters for the XRAI saliency
                          method. Defaults to None.

    Raises:
        ValueError: If algorithm type is unknown (not full or fast)

    Returns:
        XRAIOutput: an object that contains the output of the XRAI algorithm.

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
    attr = _attr_aggregation_max(attr)

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
            min_pixel_diff=1,
            integer_segments=True):
    """[summary]

    Args:
        attr: Source attributions for XRAI. XRAI attributions will be same size
              as the input attr.
        segs: Input segments as a list of boolean masks. XRAI uses these to
              compute attribution sums.
        gain_fun: The function that computes XRAI area attribution from source
                  attributions. Defaults to _gain_density, which calculates the
                  density of attributions of the mask (mean).
        area_perc_th: [description]. Defaults to 1.0.
        verbose: [description]. Defaults to 0.
        min_pixel_diff: Do not consider masks that have difference less than
                        this number compared to the current mask. Defaults to 1,
                        meaning remove the masks that completely overlap with
                        the current mask.
        integer_segments: See XRAIParameters. Defaults to True.

    Returns:
        [type]: [description]
    """
    """We expect attr to be 2D, XRAI shape is equal to attr shape
      Segs are list of binary masks, one per segment (pre-dilated if neeeded)
    """
    output_attr = -np.inf * np.ones(shape=attr.shape, dtype=np.float)

    n_masks = len(segs)
    current_attr_sum = 0.0
    current_area_perc = 0.0
    current_mask = np.zeros(attr.shape, dtype=bool)

    masks_trace = []
    remaining_masks = {ind: mask for ind, mask in enumerate(segs)}

    added_masks_cnt = 1
    # While the mask area is less than area_th and remaining_masks is not empty
    while current_area_perc <= area_perc_th:
      best_gain = -np.inf
      best_key = None
      remove_key_queue = []
      for mask_key in remaining_masks:
        mask = remaining_masks[mask_key]
        # If mask does not add more than min_pixel_diff to current mask, remove
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
      mask_diff = _get_diff_mask(added_mask, current_mask)
      masks_trace.append((mask_diff, best_gain))

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

    uncomputed_mask = output_attr == -np.inf
    # Assign the uncomputed areas a value such that sum is same as ig
    output_attr[uncomputed_mask] = gain_fun(uncomputed_mask, attr)
    # Set uncomputed region's rank to max rank + 1
    masks_trace = zip(*sorted(masks_trace, key=lambda x: -x[1]))[0]
    if integer_segments:
      attr_ranks = np.zeros(shape=attr.shape, dtype=np.int)
      for i, mask in enumerate(masks_trace):
        attr_ranks[mask] = i + 1
      assert np.all(uncomputed_mask == (attr_ranks == 0))
      attr_ranks[uncomputed_mask] = np.max(attr_ranks) + 1
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
    """[summary]

    Args:
        attr ([type]): [description]
        segs ([type]): [description]
        gain_fun ([type], optional): [description]. Defaults to _gain_density.
        area_perc_th (float, optional): [description]. Defaults to 1.0.
        verbose (int, optional): [description]. Defaults to 0.
        integer_segments (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    """
    """We expect attr to be 2D, XRAI shape is equal to attr shape
      Segs are list of binary masks, one per segment
    """
    output_attr = -np.inf * np.ones(shape=attr.shape, dtype=np.float)

    n_masks = len(segs)
    current_mask = np.zeros(attr.shape, dtype=bool)

    masks_trace = []

    # Sort all masks based on gain, ignore overlaps
    seg_attrs = [gain_fun(seg_mask, attr) for seg_mask in segs]
    segs, seg_attrs = zip(
        *sorted(zip(segs, seg_attrs), key=lambda x: -x[1]))

    for i, added_mask in enumerate(segs):
      mask_diff = _get_diff_mask(added_mask, current_mask)
      mask_gain = gain_fun(mask_diff, attr)
      masks_trace.append((mask_diff, mask_gain))
      output_attr[mask_diff] = mask_gain
      current_mask = np.logical_or(current_mask, added_mask)
      if verbose:
        current_attr_sum = np.sum(attr[current_mask])
        current_area_perc = np.mean(current_mask)
        logging.info("{} of {} masks added,"
                     "attr_sum: {}, area: {:.3g}/{:.3g}".format(
                         i + 1, n_masks, current_attr_sum, current_area_perc,
                         area_perc_th))
    masks_trace = zip(*sorted(masks_trace, key=lambda x: -x[1]))[0]
    if integer_segments:
      attr_ranks = np.zeros(shape=attr.shape, dtype=np.int)
      for i, mask in enumerate(masks_trace):
        attr_ranks[mask] = i + 1
      return output_attr, attr_ranks
    else:
      return output_attr, masks_trace

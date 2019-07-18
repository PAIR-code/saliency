"""Implementation of segment integrated gradients code.

This code is based on the structure of third_party/py/saliency.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

import numpy as np
import saliency
from skimage import segmentation
from skimage.morphology import dilation
from skimage.morphology import disk
from skimage.transform import resize

FLAGS = flags.FLAGS
# TODO (tolgab) change all prints to log


def _normalize_image(im, value_range, resize_shape=None):
  im_max = np.max(im)
  im_min = np.min(im)
  im = (im - im_min) / (im_max - im_min)
  im -= 0.5
  im *= value_range[1] - value_range[0]
  im += np.mean(value_range)
  if resize_shape is not None:
    im = resize(im, resize_shape, order=3, mode='constant',
                preserve_range=True, anti_aliasing=True)
  return im


def _get_segments_felsenschwab(im, resize_image=True,
                               scale_range=[-1.0, 1.0], dilation_rad=5):
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
      seg = segmentation.felzenszwalb(
          im, scale=scale, sigma=sigma, min_size=20)
      if resize_image:
        seg = resize(seg, original_shape, order=0, preserve_range=True,
                     mode='constant', anti_aliasing=False).astype(np.uint8)
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


def _get_diff_cnt(mask1, mask2):
  return (np.sum(np.logical_and(mask1, mask2))/np.min([np.sum(mask1), np.sum(mask2)]))


def _unpack_segs_to_masks(segs):
  masks = []
  for seg in segs:
    for l in xrange(seg.min(), seg.max()+1):
      masks.append(seg == l)
  return masks


class XRAIConfig():
  def __init__(self,
               steps=100,
               verbosity=0):
    # Number of steps to compute integrated gradients, more is slower but better
    self.steps = steps
    # # Automatically resize baseline image to fit the input image size
    # self.baseline_auto_resize = baseline_auto_resize
    self.return_baseline_predictions = False
    self.return_ig_attributions = False
    self.return_ig_for_every_step = False
    self.return_xrai_segments = False
    self.flatten_xrai_segments = True
    # XRAI algorithm.
    self.algorithm = 'full'
    # Verbosity to print status as segments are added
    self.verbosity = verbosity


class SaliencyOutput:
  def __init__(self, attribution_mask):
    self.attribution_mask = attribution_mask


class XRAIOutput(SaliencyOutput):
  def __init__(self, attribution_mask):
    super(XRAIOutput, self).__init__(attribution_mask)
    self.baselines = None
    self.error = None
    self.baseline_predictions = None
    self.ig_attribution = None
    self.segments = None


class XRAI(saliency.GradientSaliency):
  def __init__(self, graph, session, y, images):
    # Initialize integrated gradients
    self._integrated_gradients = saliency.IntegratedGradients(graph, session, y,
                                                              images)

  def _get_integrated_gradients(self, im, feed_dict, baselines, steps):
    """ Takes mean of attributions from all baselines
    """
    grads = []
    for baseline in baselines:
      grads.append(self._integrated_gradients.GetMask(
          im, feed_dict=feed_dict, x_baseline=baseline, x_steps=steps))
    return grads

  def _make_baselines(self, x_value, x_baselines):
    # If baseline is not provided default to im min and max values
    if x_baselines is None:
      x_baselines = []
      x_baselines.append(np.min(x_value)*np.ones_like(x_value))
      x_baselines.append(np.max(x_value)*np.ones_like(x_value))
    else:
      for baseline in x_baselines:
        if baseline.shape != x_value.shape:
          raise ValueError("Baseline size {} does not match input size {}".format(
              baseline.shape, x_value.shape))
    return x_baselines

  def GetMask(self, x_value, feed_dict={}, baselines=None, segments=None, extra_parameters=None):
    """ Output a np.ndarray heatmap of XRAI attributions with input shape.
    """
    results = self.GetMaskWithDetails(x_value,
                                      feed_dict=feed_dict,
                                      baselines=baselines,
                                      segments=segments,
                                      extra_parameters=extra_parameters)
    return results.attribution_mask

  def GetMaskWithDetails(self, x_value, feed_dict={}, baselines=None, segments=None, extra_parameters=None):
    """ Applies XRAI method on an input image and returns the result saliency
        mask along with other detailed information.
    Parameters:
    Returns:
      A XraiOutput object that contains the output of the XRAI algorithm.
    """
    x_baselines = self._make_baselines(x_value, baselines)

    attrs = self._get_integrated_gradients(x_value, feed_dict=feed_dict,
                                           baselines=x_baselines,
                                           steps=extra_parameters.steps)
    # Merge attributions from different baselines
    attr = np.mean(attrs, axis=0)
    # Merge attribution channels for XRAI input
    attr = _accumulate_attr_max(attr)

    if segments is not None:
      segs = segments
    else:
      segs = _get_segments_felsenschwab(x_value)

    if extra_parameters.algorithm == 'full':
      attr_map, attr_data = self.xrai(attr=attr, segs=segs,
                                      max_area_th=extra_parameters.max_area,
                                      gain_fun=_gain_density,
                                      verbose=extra_parameters.verbosity,
                                      integer_segments=extra_parameters.flatten_xrai_segments)
    elif extra_parameters.algorithm == 'fast':
      attr_map, attr_data = self.xrai_fast(attr=attr, segs=segs, gain_fun=_gain_density, verbose=0,
                                           integer_segments=extra_parameters.flatten_xrai_segments)
    else:
      print('Unknown algorithm type: {}'.format(extra_parameters.algorithm))

    results = XRAIOutput(attr_map)
    results.baselines = x_baselines
    if extra_parameters.return_xrai_segments:
      results.segments = attr_data
    if extra_parameters.return_baseline_predictions:
      baseline_predictions = []
      for baseline in x_baselines:
        baseline_predictions.append(self.predict(baseline))
      results.baseline_predictions = baseline_predictions
    if extra_parameters.ig_attributions:
      results.ig_attribution = attr
    return results

  @staticmethod
  def _xrai(attr, segs, area_perc_th,
            gain_fun, verbose=0, max_iou=0.9,
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
        mask_iou = _get_diff_cnt(mask, current_mask)
        if mask_iou >= max_iou:
          remove_key_queue.append(mask_key)
          if verbose > 2:
            print("Skipping mask with iou: {:.3g},".format(mask_iou))
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
        print("{} of {} masks added,"
              "attr_sum: {}, area: {:.3g}/{:.3g}, {} remaining masks".format(added_masks_cnt,
                                                                             n_masks, current_attr_sum, current_area_perc,
                                                                             area_perc_th, len(remaining_masks)))
      added_masks_cnt += 1

    ig_sum = np.sum(attr)
    uncomputed_mask = output_attr == -np.inf
    assert uncomputed_mask == (attr_ranks == 0)
    # Assign the uncomputed areas a value such that sum is same as ig
    output_attr[uncomputed_mask] = (
        ig_sum - current_attr_sum) / np.sum(uncomputed_mask)
    # Set uncomputed region's ranking to max rank + 1
    attr_ranks[uncomputed_mask] = np.max(attr_ranks) + 1
    if integer_segments:
      return output_attr, attr_ranks
    else:
      return output_attr, masks_trace

  @staticmethod
  def _xrai_fast(attr, segs, gain_fun, area_perc_th=1.0, verbose=0,
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
    segs = segs[sorted_inds]

    for i, added_mask in enumerate(segs):
      mask_diff = np.logical_and(np.logical_not(current_mask), added_mask)
      if not integer_segments:
        masks_trace.append(added_mask)
      else:
        attr_ranks[mask_diff] = i+1
      current_mask = np.logical_or(current_mask, added_mask)
      current_attr_sum = np.sum(attr[current_mask])
      current_area_perc = np.mean(current_mask)
      output_attr[mask_diff] = sorted_sums[i]
      if verbose:
        print("{} of {} masks added,"
              "attr_sum: {}, area: {:.3g}/{:.3g}".format(i,
                                                         n_masks, current_attr_sum, current_area_perc,
                                                         area_perc_th))
    if integer_segments:
      return output_attr, attr_ranks
    else:
      return output_attr, masks_trace

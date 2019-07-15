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


def normalize_image(im, value_range, resize_shape=None):
  im_max = np.max(im)
  im_min = np.min(im)
  im = (im - im_min) / (im_max - im_min)
  im -= 0.5
  im *= value_range[1] - value_range[0]
  im += np.mean(value_range)
  if resize_shape is not None:
    im = resize(im, resize_shape, order=3, mode='constant', preserve_range=True, anti_aliasing=True)
  return im


def get_segments_felsenschwab(im, resize_image=True,
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
    im = normalize_image(im, scale_range, (224, 224))
  else:
    im = normalize_image(im, scale_range)
  segs = []
  for scale in SCALE_VALUES:
    for sigma in SIGMA_VALUES:
      seg = segmentation.felzenszwalb(
          im, scale=scale, sigma=sigma, min_size=20)
      if resize_image:
        seg = resize(seg, original_shape, order=0, preserve_range=True, mode='constant', anti_aliasing=False).astype(np.uint8)
      segs.append(seg)
  masks = unpack_segs_to_masks(segs)
  if dilation_rad:
    selem = disk(dilation_rad)
    masks = [dilation(mask, selem=selem) for mask in masks]
  return masks


def accumulate_attr_max(attr, axis=1):
  return attr.max(axis=axis)


def gain_density(mask1, attr, mask2=None):
  # Compute the attr density over mask1. If mask2 is specified, compute density
  # for mask1 \ mask2
  if mask2 is None:
    added_mask = mask1
  else:
    added_mask = np.logical_and(mask1, np.logical_not(mask2))
  return attr[added_mask].mean()


def get_iou(mask1, mask2):
  return (np.sum(np.logical_and(mask1, mask2)) /
          np.sum(np.logical_or(mask1, mask2)))


def unpack_segs_to_masks(segs):
  masks = []
  for seg in segs:
    for l in xrange(seg.min(), seg.max()+1):
      masks.append(seg == l)
  return masks


class XRAIConfig():
  def __init__(self,
               baselines=None,
               baseline_auto_resize=True,
               steps=100,
               verbosity=0):
    # Number of steps to compute integrated gradients, more is slower but better
    self.steps = steps
    # A list of baselines to compute integrated gradients
    # Needs to match the input image size if auto_resize is disabled
    # If None, min and max values of the input image will be used
    self.baselines = baselines
    # Automatically resize baseline image to fit the input image size
    self.baseline_auto_resize = baseline_auto_resize
    self.return_baseline_predictions = False
    self.return_ig_attributions = False
    self.return_ig_for_every_step = False
    self.return_xrai_segments = False
    self.flatten_xrai_segments=True
    # Specifies a flavour of the XRAI algorithm. ‘full’ - executes slower but
    # more precise XRAI algorithm. ‘fast’ - executes faster but less precise
    # XRAI algorithm.
    self.algorithm = 'full'
    # Verbosity to print status as segments are added
    self.verbosity = verbosity


class SegmentIntegratedGradients(saliency.GradientSaliency):
  def __init__(self, graph, session, y, images):
    # Initialize integrated gradients
    self.integrated_gradients = saliency.IntegratedGradients(graph, session, y,
                                                             images)

  def get_integrated_gradients_base(self, im, feed_dict, baseline, steps):
    grads = self.integrated_gradients.GetMask(
      im, feed_dict = feed_dict, x_baseline=baseline, x_steps=steps)
    return grads

  def get_integrated_gradients_mean(self, im, feed_dict, baselines, steps):
    """ Takes mean of attributions from all baselines
    """
    grads = []
    for baseline in baselines:
      grads.append(self.get_integrated_gradients_base(im, feed_dict, baseline, steps))
    return np.mean(grads, axis=0)

  def make_baselines(self, x_value, sig_config):
    x_baselines = sig_config.baselines
    # If baseline is not provided default to im min and max values
    if x_baselines is None:
      x_baselines = []
      x_baselines.append(np.min(x_value)*np.ones_like(x_value))
      x_baselines.append(np.max(x_value)*np.ones_like(x_value))
    else:
      for i, baseline in enumerate(x_baselines):
        if baseline.shape != x_value.shape:
          if sig_config.baseline_auto_resize:
            baseline = resize(baseline, x_value.shape, preserve_range=True)
            x_baselines[i] = baseline
          else:
            raise ValueError("Baseline size {} does not match input size {}".format(baseline.shape, x_value.shape))
    return x_baselines

  def GetMask(self, x_value, sig_config, feed_dict={}, max_area=1.0, segments=None):
    """ This outputs a heatmap of size of the input image with SIG attributions

    output is 3D with third dimension = 1
    """
    x_baselines = self.make_baselines(x_value, sig_config)

    attr = self.get_integrated_gradients_mean(x_value, feed_dict=feed_dict,
                                              baselines=x_baselines,
                                              steps=sig_config.steps)
    segs = sig_config.segmentation_fun(x_value)
    (percent_masks, masks_trace) = self.sig(im=x_value, attr=attr, segs=segs, percent_areas=[max_area], verbose=sig_config.verbosity)
    # Unpack masks
    # TODO(tolgab) Directly return float heatmap instead of unpacking from trace
    sig_attr_raw = -np.inf * np.ones(shape=x_value.shape[:2], dtype=np.float)
    # masks_trace : current_mask, added_mask, current_attr_sum, current_area_perc
    for ii in xrange(1, len(masks_trace)):
      mask_diff = np.logical_and(np.logical_not(masks_trace[ii-1][0]), masks_trace[ii][0])
      if np.sum(mask_diff) == 0:
        continue
      sig_attr_raw[mask_diff] = calculate_attr_max(mask_diff, attr)
    sig_attr_raw[sig_attr_raw==-np.inf] = np.min(sig_attr_raw[sig_attr_raw!=-np.inf]) - 0.1

    return sig_attr_raw


@staticmethod
def _xrai(attr, segs, area_perc_th,
    gain_fun, verbose=0, max_iou=1.0,
    integer_segments=True):
  """We expect attr to be 2D, SIG shape is equal to attr shape
    Segs are list of binary masks, one per segment (pre-dilated if neeeded)
  """
  sig_attr_raw = -np.inf * np.ones(shape=attr.shape, dtype=np.float)

  n_masks = len(segs)
  current_attr_sum = 0.0
  current_area_perc = 0.0
  current_mask = np.zeros(attr.shape, dtype=bool)

  masks_trace = []
  attr_ranks = np.array(shape=attr.shape, dtype=np.int)
  remaining_masks = {ind: mask for ind, mask in enumerate(segs)}

  added_masks_cnt = 0
  # While the mask area is less than area_th and remaining_masks is not empty
  while remaining_masks and current_area_perc <= area_perc_th:
    best_gain = -np.inf
    best_key = None
    for mask_key, mask in remaining_masks.iteritems():
      # If mask overlaps current mask more than max_iou then delete it
      mask_iou = get_iou(mask, current_mask)
      if mask_iou > max_iou:
        del remaining_masks[mask_key]
        if verbose > 1:
          print("Skipping mask with iou: {:.3g},".format(mask_iou))
        continue
      gain = gain_fun(current_mask, attr, mask2=mask)
      if gain > best_gain:
        best_gain = gain
        best_key = mask_key

    added_mask = remaining_masks[best_key]
    mask_diff = np.logical_and(np.logical_not(current_mask), added_mask)
    if not integer_segments:
      masks_trace.append(added_mask)
    else:
      attr_ranks[mask_diff] = added_masks_cnt
    current_mask = np.logical_or(current_mask, added_mask)
    current_attr_sum = np.sum(attr[current_mask])
    current_area_perc = np.mean(current_mask)
    sig_attr_raw[mask_diff] = gain
    del remaining_masks[best_key]  # delete used key
    if verbose:
      print("{} of {} masks added,"
            "attr_sum: {}, area: {:.3g}/{:.3g}".format(n_masks-len(remaining_masks),
                                        n_masks, current_attr_sum, current_area_perc,
                                        area_perc_th))
  if integer_segments:
    return sig_attr_raw, attr_ranks
  else:
    return sig_attr_raw, masks_trace


@staticmethod
def _xrai_fast(attr, segs, area_perc_th,
    gain_fun, verbose=0, max_iou=1.0,
    integer_segments=True):
  """We expect attr to be 2D, SIG shape is equal to attr shape
    Segs are list of binary masks, one per segment (pre-dilated if neeeded)
  """
  sig_attr_raw = -np.inf * np.ones(shape=attr.shape, dtype=np.float)

  n_masks = len(segs)
  current_attr_sum = 0.0
  current_area_perc = 0.0
  current_mask = np.zeros(attr.shape, dtype=bool)

  masks_trace = []
  attr_ranks = np.array(shape=attr.shape, dtype=np.int)

  # Sort all masks based on gain, ignore overlaps
  attr_sums = map(gain_fun, segs)
  sorted_inds, sorted_sums = sorted(zip(range(n_masks), attr_sums), key=lambda x: x[1])
  segs = segs[sorted_inds]

  # While the mask area is less than area_th and remaining_masks is not empty
  for i, added_mask in enumerate(segs):
    # and current_area_perc <= area_perc_th:

    mask_diff = np.logical_and(np.logical_not(current_mask), added_mask)
    if not integer_segments:
      masks_trace.append(added_mask)
    else:
      attr_ranks[mask_diff] = i
    current_mask = np.logical_or(current_mask, added_mask)
    current_attr_sum = np.sum(attr[current_mask])
    current_area_perc = np.mean(current_mask)
    sig_attr_raw[mask_diff] = sorted_sums[i]
    if verbose:
      print("{} of {} masks added,"
            "attr_sum: {}, area: {:.3g}/{:.3g}".format(i,
                                        n_masks, current_attr_sum, current_area_perc,
                                        area_perc_th))
  if integer_segments:
    return sig_attr_raw, attr_ranks
  else:
    return sig_attr_raw, masks_trace




  # def GetMask(self, x_value, sig_config, feed_dict={}, percent_areas=[0.1]):
  #   x_baselines = sig_config.baselines
  #   # If baseline is not provided default to im min and max values
  #   if x_baselines is None:
  #     x_baselines = []
  #     x_baselines.append(np.min(x_value)*np.ones_like(x_value))
  #     x_baselines.append(np.max(x_value)*np.ones_like(x_value))
  #   else:
  #     for i, baseline in enumerate(x_baselines):
  #       if baseline.shape != x_value.shape:
  #         if sig_config.baseline_auto_resize:
  #           baseline = resize(baseline, x_value.shape, preserve_range=True)
  #           x_baselines[i] = baseline
  #         else:
  #           raise ValueError("Baseline size {} does not match input size {}".format(baseline.shape, x_value.shape))

  #   attr = self.get_integrated_gradients_mean(x_value, feed_dict=feed_dict,
  #                                             baselines=x_baselines,
  #                                             steps=sig_config.steps)
  #   segs = sig_config.segmentation_fun(x_value)
  #   (percent_masks, masks_trace) = self.sig(im=x_value, attr=attr, segs=segs, percent_areas=percent_areas, verbose=sig_config.verbosity)
  #   return percent_masks, masks_trace

  # @classmethod
  # def _sig_heatmap(cls, im, attr, max_area=1.0, verbose=0):
  #   segs = get_segments_felsenschwab(im)
  #   # Work in progress
  #   (percent_masks, masks_trace) = cls.sig(im=im, attr=attr, segs=segs, percent_areas=[max_area], verbose=verbose)
  #   # Unpack masks
  #   # TODO(tolgab) seperate baseline generation from mask generation
  #   # TODO(tolgab) Directly return float heatmap instead of unpacking from trace
  #   sig_attr_raw = -np.inf * np.ones(shape=im.shape[:2], dtype=np.float)
  #   # masks_trace : current_mask, added_mask, current_attr_sum, current_area_perc
  #   for ii in xrange(1, len(masks_trace)):
  #     mask_diff = np.logical_and(np.logical_not(masks_trace[ii-1][0]), masks_trace[ii][0])
  #     if np.sum(mask_diff) == 0:
  #       continue
  #     sig_attr_raw[mask_diff] = calculate_attr_max(mask_diff, attr)
  #   sig_attr_raw[sig_attr_raw==-np.inf] = np.min(sig_attr_raw[sig_attr_raw!=-np.inf]) - 0.1

  #   return sig_attr_raw

  # @staticmethod
  # def sig(im, attr, segs, percent_areas, dilation_rad=5,
  #     gain_fun=calculate_attr_max, verbose=0):

  #   # Compute superpixels
  #   all_masks = unpack_segs_to_masks(segs)
  #   if dilation_rad:
  #     selem = disk(dilation_rad)
  #     all_masks = [dilation(mask, selem=selem) for mask in all_masks]

  #   percent_areas = np.array(percent_areas, dtype=float)
  #   n_masks = len(all_masks)
  #   area_perc_th = np.max(percent_areas)
  #   current_attr_sum = 0.0
  #   current_area_perc = 0.0
  #   current_mask = np.zeros(im.shape[:2], dtype=bool)
  #   masks_trace = [[current_mask, current_mask, current_attr_sum, current_area_perc]]
  #   percent_masks = np.array([None] * len(percent_areas), dtype=object)
  #   remaining_masks = {ind: mask for ind, mask in enumerate(all_masks)}
  #   # While the mask area is less than area_th and remaining_masks is not empty
  #   while remaining_masks and current_area_perc <= area_perc_th:
  #     best_gain = -np.inf
  #     best_key = None
  #     for mask_key, mask in remaining_masks.iteritems():
  #       gain = gain_fun(current_mask, attr, mask2=mask)
  #       if gain > best_gain:
  #         best_gain = gain
  #         best_key = mask_key
  #     # added_mask = dilation(remaining_masks[best_key], selem=selem)  # for historical reasons, remove after testing
  #     added_mask = remaining_masks[best_key]
  #     current_mask = np.logical_or(current_mask, added_mask)
  #     current_attr_sum = best_gain
  #     current_area_perc = np.mean(current_mask)
  #     # TODO(tolgab) Change arrays to namedtuples
  #     masks_trace.append([current_mask, added_mask, current_attr_sum, current_area_perc])
  #     del remaining_masks[best_key]  # delete used key
  #     if verbose:
  #       print("{} of {} masks added,"
  #             "area: {:.3g}/{:.3g}".format(n_masks-len(remaining_masks),
  #                                         n_masks, current_area_perc,
  #                                         area_perc_th))
  #     area_check = (current_area_perc > percent_areas) & (percent_masks == None)
  #     for ind in np.where(area_check)[0]:
  #       percent_masks[ind] = masks_trace[-2][:1] + [percent_areas[ind]]

  #   if percent_areas[-1] == 1.0 and percent_masks[-1] is None:
  #     percent_masks[-1] = [np.ones(shape=current_mask.shape, dtype=np.bool), 1.0]

  #   return percent_masks, masks_trace

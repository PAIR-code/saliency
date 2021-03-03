import mock
import numpy as np
import skimage.draw as sk_draw
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.python.platform import googletest
from . import xrai
from .xrai import XRAI, XRAIParameters

IMAGE_SIZE = 299


class XraiTest(googletest.TestCase):
  """
  To run:
  "python -m saliency.xrai_test" from the PAIR-code/saliency directory.
  """

  def setUp(self):
    # Mock IntegratedGradients.
    self.mock_ig = mock.patch(__name__ + '.xrai.IntegratedGradients').start()
    self.input_image = np.random.rand(IMAGE_SIZE, IMAGE_SIZE, 3) * 0.3 + 0.5
    self.ig_bl_1_attr = np.random.rand(IMAGE_SIZE, IMAGE_SIZE, 3) - 0.4
    self.ig_bl_2_attr = np.random.rand(IMAGE_SIZE, IMAGE_SIZE, 3) - 0.4
    self.mock_ig_instance = mock.Mock()
    self.mock_ig_instance.GetMask.side_effect = [self.ig_bl_1_attr,
                                                 self.ig_bl_2_attr,
                                                 self.ig_bl_1_attr,
                                                 self.ig_bl_2_attr]
    self.mock_ig.return_value = self.mock_ig_instance

    # IG attribution that XRAI should calculate internally.
    self.ig_mask = np.asarray([self.ig_bl_1_attr, self.ig_bl_2_attr]).mean(
        axis=0)
    # A mocked XRAI object that uses neither the graph, session, x or y values.
    self.xrai = XRAI(tf.Graph(), tf.Session(), y=np.array([0]), x=np.array([0]))

  def tearDown(self):
    self.mock_ig.stop()

  def testXraiGetMaskFullFlat(self):
    # Calculate XRAI attribution using GetMaskWithDetails(...) method.
    xrai_params = XRAIParameters(return_xrai_segments=True,
                                 flatten_xrai_segments=True,
                                 algorithm='full',
                                 area_threshold=1.0,
                                 return_ig_attributions=True)
    xrai_out = self.xrai.GetMaskWithDetails(x_value=self.input_image,
                                            extra_parameters=xrai_params)

    # Verify the result.
    self._assert_xrai_correctness(xrai_out, is_flatten_segments=True)

    # Calculate XRAI attribution using GetMask(...) method.
    heatmap = self.xrai.GetMask(x_value=self.input_image,
                                extra_parameters=xrai_params)

    # Verify that the heatmaps returned by GetMaskWithDetails(...) and
    # GetMaskWithDetails(...) are the same.
    self.assertTrue(np.array_equal(xrai_out.attribution_mask, heatmap))

  def testXraiGetMaskFastFlat(self):
    # Calculate XRAI attribution using GetMaskWithDetails(...) method.
    xrai_params = XRAIParameters(return_xrai_segments=True,
                                 flatten_xrai_segments=True,
                                 algorithm='fast',
                                 return_ig_attributions=True)
    xrai_out = self.xrai.GetMaskWithDetails(x_value=self.input_image,
                                            extra_parameters=xrai_params)

    # Verify the result.
    self._assert_xrai_correctness(xrai_out, is_flatten_segments=True)

    # Calculate XRAI attribution using GetMask(...) method.
    heatmap = self.xrai.GetMask(x_value=self.input_image,
                                extra_parameters=xrai_params)

    # Verify that the heatmaps returned by GetMaskWithDetails(...) and
    # GetMaskWithDetails(...) are the same.
    self.assertTrue(np.array_equal(xrai_out.attribution_mask, heatmap))

  def testXraiGetMaskFullNonFlat(self):
    # Calculate XRAI attribution using GetMaskWithDetails(...) method.
    xrai_params = XRAIParameters(return_xrai_segments=True,
                                 flatten_xrai_segments=False,
                                 area_threshold=1.0,
                                 algorithm='full',
                                 return_ig_attributions=True)
    xrai_out = self.xrai.GetMaskWithDetails(x_value=self.input_image,
                                            extra_parameters=xrai_params)

    # Verify the result.
    self._assert_xrai_correctness(xrai_out, is_flatten_segments=False)

    # Calculate XRAI attribution using GetMask(...) method.
    heatmap = self.xrai.GetMask(x_value=self.input_image,
                                extra_parameters=xrai_params)

    # Verify that the heatmaps returned by GetMaskWithDetails(...) and
    # GetMaskWithDetails(...) are the same.
    self.assertTrue(np.array_equal(xrai_out.attribution_mask, heatmap))

  def testXraiGetMaskFastNonFlat(self):
    # Calculate XRAI attribution using GetMaskWithDetails(...) method.
    xrai_params = XRAIParameters(return_xrai_segments=True,
                                 flatten_xrai_segments=False,
                                 algorithm='fast',
                                 return_ig_attributions=True)
    xrai_out = self.xrai.GetMaskWithDetails(x_value=self.input_image,
                                            extra_parameters=xrai_params)
    # Verify the result.
    self._assert_xrai_correctness(xrai_out, is_flatten_segments=False)

    # Calculate XRAI attribution using GetMask(...) method.
    heatmap = self.xrai.GetMask(x_value=self.input_image,
                                extra_parameters=xrai_params)

    # Verify that the heatmaps returned by GetMaskWithDetails(...) and
    # GetMaskWithDetails(...) are the same.
    self.assertTrue(np.array_equal(xrai_out.attribution_mask, heatmap))

  def testCustomSegments(self):
    # Create the first segment.
    rec_1 = sk_draw.rectangle(start=(10, 10), end=(30, 30),
                              shape=(IMAGE_SIZE, IMAGE_SIZE))
    seg_1 = np.zeros(shape=(IMAGE_SIZE, IMAGE_SIZE), dtype=np.bool)
    seg_1[tuple(rec_1)] = True

    # Create the second segment.
    rec_2 = sk_draw.rectangle(start=(60, 60), end=(100, 100),
                              shape=(IMAGE_SIZE, IMAGE_SIZE))
    seg_2 = np.zeros(shape=(IMAGE_SIZE, IMAGE_SIZE), dtype=np.bool)
    seg_2[tuple(rec_2)] = True

    # Calculate the XRAI attribution.
    xrai_params = XRAIParameters(return_xrai_segments=True,
                                 flatten_xrai_segments=False,
                                 area_threshold=1.0,
                                 return_ig_attributions=True)
    xrai_out = self.xrai.GetMaskWithDetails(x_value=self.input_image,
                                            extra_parameters=xrai_params,
                                            segments=[seg_1, seg_2])

    # Verify correctness of the attribution.
    self._assert_xrai_correctness(xrai_out, is_flatten_segments=False)

    # Verify that the segments are ordered correctly, i.e. the segment with
    # higher attribution comes first.
    seg1_attr = self.ig_mask[seg_1].max(axis=1).mean()
    seg2_attr = self.ig_mask[seg_2].max(axis=1).mean()
    if seg1_attr >= seg2_attr:
      self.assertTrue(np.array_equal(seg_1, xrai_out.segments[0]),
                      msg='The segments might be ordered incorrectly.')
    else:
      self.assertTrue(np.array_equal(seg_2, xrai_out.segments[0]),
                      msg='The segments might be ordered incorrectly.')

    # Three segments are expected. The last segment should include all pixels
    # that weren't included in the first two segments.
    self.assertEqual(3, len(xrai_out.segments),
                     'Unexpected the number of returned segments.')

  def testBaselines(self):
    """Tests that a client can pass an arbitrary baseline values; and that
       these baselines are actually used for calculating the Integrated
       Gradient masks.
    """
    # Create baselines and pass them to XRAI.GetMask(...).
    baseline_1 = np.random.rand(IMAGE_SIZE, IMAGE_SIZE, 3)
    baseline_2 = np.random.rand(IMAGE_SIZE, IMAGE_SIZE, 3)
    self.xrai.GetMask(x_value=self.input_image,
                      baselines=[baseline_1, baseline_2])

    # Verify that the XRAI object called Integrated Gradients with the baselines
    # that were passed to xrai.GetMask(...).
    calls = self.mock_ig_instance.method_calls
    self.assertEqual(2, len(calls),
                     'There should be only two calls to IG Getmask()')
    self.assertTrue(np.array_equal(baseline_1, calls[0][2]['x_baseline']),
                    msg='IG was called with incorrect baseline.')
    self.assertTrue(np.array_equal(baseline_2, calls[1][2]['x_baseline']),
                    msg='IG was called with incorrect baseline.')

  def testBaseAttribution(self):
    base_attribution = np.random.rand(IMAGE_SIZE, IMAGE_SIZE, 3)

    # Calculate XRAI attribution using GetMask(...) method.
    heatmap = self.xrai.GetMask(x_value=self.input_image,
                                base_attribution=base_attribution)
    # Make sure that the GetMask() method doesn't return the same attribution
    # that was passed to it.
    self.assertFalse(np.array_equal(base_attribution.max(axis=2), heatmap))
    # The sum of XRAI attribution should be equal to the sum of the underlying
    # base attribution. Internally the attribution that is used by XRAI is
    # the max over color channels.
    self.assertAlmostEqual(base_attribution.max(axis=2).sum(), heatmap.sum())

    # Verify that the XRAI object didn't called Integrated Gradients.
    calls = self.mock_ig_instance.method_calls
    self.assertEqual(0, len(calls),
                     'XRAI should not call Integrated Gradients.')

  def testBaseAttributionMismatchedShape(self):
    # Create base_attribution that shape doesn't match the input.
    base_attribution = np.random.rand(IMAGE_SIZE, IMAGE_SIZE + 1, 3)

    # Verify that the exception was raised.
    with self.assertRaisesRegexp(ValueError, 'The base attribution shape '
                                             'should'):
      # Calling GetMask(...) should result in exception.
      self.xrai.GetMask(x_value=self.input_image,
                        base_attribution=base_attribution)

  def _assert_xrai_correctness(self, xrai_out, is_flatten_segments):
    """Performs general XRAIOutput verification that is applicable for all
       XRAI results.
    """
    xrai_attribution_mask = xrai_out.attribution_mask
    xrai_segments = xrai_out.segments

    # Check completeness with respect to IG attribution.
    self.assertAlmostEqual(self.ig_mask.max(axis=2).sum(),
                           xrai_attribution_mask.sum(),
                           msg='The sum of IG attribution (max along the color '
                               'axis) should be equal to the sum of XRAI '
                               'attribution.')

    # Check that the returned IG values are the same as returned by IG.
    self.assertTrue(
        np.array_equal(self.ig_bl_1_attr, xrai_out.ig_attribution[0]),
        msg='IG values returned by IG and returned to the client do not match.')
    self.assertTrue(
        np.array_equal(self.ig_bl_2_attr, xrai_out.ig_attribution[1]),
        msg='IG values returned by IG and returned to the client do not match.')

    # If the result is flattened, verify that the first segment is assigned
    # value 1. Convert the flattened integer segments to individual boolean
    # segments.
    segment_masks = []
    if is_flatten_segments:
      first_segment_id = xrai_segments.min()
      last_segment_id = xrai_segments.max()
      self.assertEqual(1, first_segment_id, msg='The first segment should'
                                                ' be assigned value "1".')
      for segment_id in range(first_segment_id, last_segment_id + 1):
        segment_masks.append(xrai_segments == segment_id)
    else:
      segment_masks = xrai_segments

    # Verify that 1) the segments are ordered according to their attribution;
    # 2) that every segment preserves the completeness properties;
    # 3) all pixels within a single segment have the same attribution.
    prev_seg_attr = np.inf
    for i, segment_mask in enumerate(segment_masks):
      segment_id = i + 1
      segment_attr = xrai_attribution_mask[segment_mask]
      self.assertGreater(segment_mask.sum(), 0,
                         msg='Segment {} of {} has zero area.'.format(
                             segment_id, len(segment_masks)))
      self.assertEqual(segment_attr.min(), segment_attr.max(),
                       'All attribution values within a single segment should '
                       'be equal.')
      segment_attr = segment_attr.max()
      self.assertAlmostEqual(self.ig_mask.max(axis=2)[segment_mask].sum(),
                             xrai_attribution_mask[segment_mask].sum(),
                             msg='The sum of the XRAI attribution within a '
                                 'segment should be equal to the sum of IG '
                                 'attribution within the same segment.')
      # The last segment may have attribution higher than the previous one
      # because it includes all pixels that weren't included by the previous
      # segments.
      if i < len(segment_masks) - 1:
        self.assertLessEqual(segment_attr, prev_seg_attr,
                             'Pixel attributions of a segment with higher id '
                             'should be lower than pixel attributions of a '
                             'segment with a lower id. Segment {}'.format(
                                 segment_id))
      prev_seg_attr = segment_attr


if __name__ == '__main__':
  googletest.main()

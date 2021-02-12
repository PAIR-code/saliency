from ..core import xrai as core_xrai
from .base import TF1CoreSaliency

XRAIParameters = core_xrai.XRAIParameters

class XRAI(TF1CoreSaliency):
  r"""A SaliencyMask class that computes saliency masks with a gradient."""

  def __init__(self, graph, session, y, x):
    super(XRAI, self).__init__(graph, session, y, x)
    self.core_instance = core_xrai.XRAI()

  def GetMask(self,
              x_value,
              feed_dict={},
              baselines=None,
              segments=None,
              base_attribution=None,
              batch_size=1,
              extra_parameters=None):
    """ Applies XRAI method on an input image and returns the result saliency
    heatmap.


    Args:
        x_value: Input ndarray.
        call_model_function: A function that interfaces with a model to return
          specific data in a dictionary when given an input and other arguments.
          Expected function signature:
          - call_model_function(x_value_batch,
                                call_model_args=None,
                                expected_keys=None):
            x_value_batch - Input for the model, given as a batch (i.e. dimension
              0 is the batch dimension, dimensions 1 through n represent a single
              input).
            call_model_args - Other arguments used to call and run the model.
            expected_keys - List of keys that are expected in the output. For 
            this method (XRAI), the expected keys are
            OUTPUT_LAYER_GRADIENTS - Gradients of the output layer (logit/softmax)
              with respect to the input. Shape should be the same shape as
              x_value_batch.
        call_model_args: The arguments that will be passed to the call model
          function, for every call of the model.
        baselines: a list of baselines to use for calculating
                   Integrated Gradients attribution. Every baseline in
                   the list should have the same dimensions as the
                   input. If the value is not set then the algorithm
                   will make the best effort to select default
                   baselines. Defaults to None.
        segments: the list of precalculated image segments that should
                  be passed to XRAI. Each element of the list is an
                  [N,M] boolean array, where NxM are the image
                  dimensions. Each elemeent on the list contains exactly the
                  mask that corresponds to one segment. If the value is None,
                  Felzenszwalb's segmentation algorithm will be applied.
                  Defaults to None.
        base_attribution: an optional pre-calculated base attribution that XRAI
                          should use. The shape of the parameter should match
                          the shape of `x_value`. If the value is None, the
                          method calculates Integrated Gradients attribution and
                          uses it.
        extra_parameters: an XRAIParameters object that specifies
                          additional parameters for the XRAI saliency
                          method. If it is None, an XRAIParameters object
                          will be created with default parameters. See
                          XRAIParameters for more details.

    Raises:
        ValueError: If algorithm type is unknown (not full or fast).
                    If the shape of `base_attribution` dosn't match the shape of
                    `x_value`.

    Returns:
        np.ndarray: A numpy array that contains the saliency heatmap.


    TODO(tolgab) Add output_selector functionality from XRAI API doc
    """
    return self.core_instance.GetMask(x_value,
        call_model_function=self.call_model_function,
        call_model_args=feed_dict,
        baselines=baselines,
        segments=segments,
        base_attribution=base_attribution,
        batch_size=batch_size,
        extra_parameters=extra_parameters)

  def GetMaskWithDetails(self, x_value, feed_dict={}, 
                         baselines=None,
                         segments=None,
                         base_attribution=None,
                         batch_size=1,
                         extra_parameters=None):
    return self.core_instance.GetMaskWithDetails(
        x_value, 
        call_model_function=self.call_model_function,
        call_model_args=feed_dict,
        baselines=baselines,
        segments=segments,
        base_attribution=base_attribution,
        batch_size=batch_size,
        extra_parameters=extra_parameters)
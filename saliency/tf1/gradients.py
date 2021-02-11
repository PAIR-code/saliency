from ..core import gradients
from .base import TF1CoreSaliency

class GradientSaliency(TF1CoreSaliency):
  r"""A SaliencyMask class that computes saliency masks with a gradient."""

  def __init__(self, graph, session, y, x):
    super(GradientSaliency, self).__init__(graph, session, y, x)
    self.core_instance = gradients.GradientSaliency()

  def GetMask(self, x_value, feed_dict={}):
    """Returns a vanilla gradient mask.

    Args:
      x_value: Input value, not batched.
      feed_dict: (Optional) feed dictionary to pass to the session.run call.
    """
    return self.core_instance.GetMask(x_value, 
        self.call_model_function,
        call_model_args=feed_dict)
from ..core import occlusion as core_occlusion
from .base import TF1CoreSaliency

class Occlusion(TF1CoreSaliency):
  r"""A SaliencyMask class that computes saliency masks with a gradient."""

  def __init__(self, graph, session, y, x):
    super(Occlusion, self).__init__(graph, session, y, x)
    self.core_instance = core_occlusion.Occlusion()

  def GetMask(self, x_value, feed_dict={}, size=15, value=0):
    """Returns a integrated gradients mask.

    Args:
      x_value: Input value, not batched.
      feed_dict: (Optional) feed dictionary to pass to the session.run call.
    """
    return self.core_instance.GetMask(x_value, 
        self.call_model_function,
        call_model_args=feed_dict,
        size=size,
        value=value)
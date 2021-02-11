from ..core import integrated_gradients
from .base import TF1CoreSaliency

class IntegratedGradients(TF1CoreSaliency):
  r"""A SaliencyMask class that computes saliency masks with a gradient."""

  def __init__(self, graph, session, y, x):
    super(IntegratedGradients, self).__init__(graph, session, y, x)
    self.core_instance = integrated_gradients.IntegratedGradients()

  def GetMask(self, x_value, feed_dict={},
              x_baseline=None, x_steps=25, batch_size=1):
    """Returns a integrated gradients mask.

    Args:
      x_value: Input value, not batched.
      feed_dict: (Optional) feed dictionary to pass to the session.run call.
    """
    return self.core_instance.GetMask(x_value, 
        self.call_model_function,
        call_model_args=feed_dict,
        x_baseline=x_baseline,
        x_steps=x_steps,
        batch_size=batch_size)
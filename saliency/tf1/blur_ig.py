from ..core import blur_ig
from .base import TF1CoreSaliency

class BlurIG(TF1CoreSaliency):
  """A F1CoreSaliency class that implements IG along blur path.

  https://arxiv.org/abs/2004.03383

  Generates a saliency mask by computing integrated gradients for a given input
  and prediction label using a path that successively blurs the image.
  """

  def __init__(self, graph, session, y, x):
    super(BlurIG, self).__init__(graph, session, y, x)
    self.core_instance = blur_ig.BlurIG()

  def GetMask(self, x_value, feed_dict={},
              max_sigma=50,
              steps=100,
              grad_step=0.01,
              sqrt=False,
              batch_size=1):
    """Returns a BlurIG mask.

    Args:
      x_value: Input value, not batched.
      feed_dict: (Optional) feed dictionary to pass to the session.run call.
    """
    return self.core_instance.GetMask(x_value, 
        self.call_model_function,
        call_model_args=feed_dict,
        max_sigma=max_sigma,
        steps=steps,
        grad_step=grad_step,
        sqrt=sqrt,
        batch_size=batch_size)
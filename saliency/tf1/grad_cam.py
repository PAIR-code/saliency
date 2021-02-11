from ..core import grad_cam
from .base import TF1CoreSaliency

class GradCam(TF1CoreSaliency):
  """A TF1CoreSaliency class that computes saliency masks with Grad-CAM.

  https://arxiv.org/abs/1610.02391

  Example usage (based on Examples.ipynb):

  grad_cam = GradCam()
  mask = grad_cam.GetMask(im,
                          call_model_function,
                          call_model_args = {neuron_selector: prediction_class},
                          should_resize = False,
                          three_dims = False)

  The Grad-CAM paper suggests using the last convolutional layer, which would
  be 'Mixed_5c' in inception_v2 and 'Mixed_7c' in inception_v3.

  """

  def __init__(self, graph, session, y=None, x=None, conv_layer=None):
    super(GradCam, self).__init__(graph, session, y, x, conv_layer)
    self.core_instance = grad_cam.GradCam()

  def GetMask(self, x_value, feed_dict={},
              should_resize=True,
              three_dims=True):
    """Returns a GradCAM mask.

    Args:
      x_value: Input value, not batched.
      feed_dict: (Optional) feed dictionary to pass to the session.run call.
    """
    return self.core_instance.GetMask(x_value, 
        self.call_model_function,
        call_model_args=feed_dict,
        should_resize=should_resize,
        three_dims=three_dims)
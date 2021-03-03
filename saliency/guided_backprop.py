# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilites to computed GuidedBackprop SaliencyMasks"""

from .base import SaliencyMask
import tensorflow.compat.v1 as tf

class GuidedBackprop(SaliencyMask):
  """A SaliencyMask class that computes saliency masks with GuidedBackProp.

  This implementation copies the TensorFlow graph to a new graph with the ReLU
  gradient overwritten as in the paper:
  https://arxiv.org/abs/1412.6806

  Thanks to Chris Olah for generously sharing his implementation of the ReLU
  backprop.
  """

  GuidedReluRegistered = False

  def __init__(self,
               graph,
               session,
               y,
               x,
               tmp_ckpt_path='/tmp/guided_backprop_ckpt'):
    """Constructs a GuidedBackprop SaliencyMask."""
    super(GuidedBackprop, self).__init__(graph, session, y, x)

    self.x = x

    if GuidedBackprop.GuidedReluRegistered is False:
      #### Acknowledgement to Chris Olah ####
      @tf.RegisterGradient("GuidedRelu")
      def _GuidedReluGrad(op, grad):
        gate_g = tf.cast(grad > 0, "float32")
        gate_y = tf.cast(op.outputs[0] > 0, "float32")
        return gate_y * gate_g * grad
    GuidedBackprop.GuidedReluRegistered = True

    with graph.as_default():
      saver = tf.train.Saver()
      saver.save(session, tmp_ckpt_path)

    graph_def = graph.as_graph_def()

    self.guided_graph = tf.Graph()
    with self.guided_graph.as_default():
      self.guided_sess = tf.Session(graph = self.guided_graph)
      with self.guided_graph.gradient_override_map({'Relu': 'GuidedRelu'}):
        # Import the graph def, and all the variables.
        tf.import_graph_def(graph_def, name='')
        saver.restore(self.guided_sess, tmp_ckpt_path)

        imported_y = self.guided_graph.get_tensor_by_name(y.name)
        imported_x = self.guided_graph.get_tensor_by_name(x.name)

        self.guided_grads_node = tf.gradients(imported_y, imported_x)[0]

  def GetMask(self, x_value, feed_dict = {}):
    """Returns a GuidedBackprop mask."""
    with self.guided_graph.as_default():
      # Move all the feed dict tensor keys to refer to the same tensor on the
      # new graph.
      guided_feed_dict = {}
      for tensor in feed_dict:
        guided_feed_dict[tensor.name] = feed_dict[tensor]
      guided_feed_dict[self.x.name] = [x_value]

    return self.guided_sess.run(
        self.guided_grads_node, feed_dict = guided_feed_dict)[0]

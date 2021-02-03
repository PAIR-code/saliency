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

"""Utilities to run CallModelSaliency methods."""
from .base import OUTPUT_GRADIENTS
from .base import CONVOLUTION_LAYER
from .base import CONVOLUTION_GRADIENTS
import tensorflow.compat.v1 as tf


def create_tf1_call_model_function(graph, session, y=None, x=None, conv_layer=None):
  """Constructs a SaliencyMask by computing dy/dx.

  Args:
    graph: The TensorFlow graph to evaluate masks on.
    session: The current TensorFlow session.
    y: The output tensor of the model. This tensor should be of size 1.
    x: The input tensor of the model. The outer dimension should be the batch 
        size.
    conv_layer: The convolution layer tensor of the model. The outer
        dimension should be the batch size.
  """
  if x is None:
    raise ValueError('Expected input tensor for x but is equal to None.')
  if y is not None:
    # y must be of size one, otherwise the gradient we get from tf.gradients
    # will be summed over all ys.
    size = 1
    for shape in y.shape:
      size *= shape
    assert size == 1
    output_gradients = tf.gradients(y, x)[0]
  if conv_layer is not None:
    conv_gradients = tf.gradients(conv_layer, x)[0]

  def convert_keys_to_fetches(expected_keys):
    fetches = []
    for expected_key in expected_keys:
      if expected_key == OUTPUT_GRADIENTS:
        if y is None:
          raise RuntimeError('Cannot return key {} because no y was specified'.format(OUTPUT_GRADIENTS))
        fetches.append(output_gradients)
      elif expected_key in [CONVOLUTION_LAYER, CONVOLUTION_GRADIENTS]:
        if conv_layer is None:
          raise RuntimeError('Cannot return key {} because no conv_layer was specified'.format(expected_key))
        if expected_key == CONVOLUTION_LAYER:
          fetches.append(conv_layer)
        else:
          fetches.append(conv_gradients)
      else:
        raise ValueError('Invalid expected key {}'.format(expected_key))
    return fetches

  def call_model_function(x_value_batch, call_model_args={}, expected_keys=None):
    call_model_args[x] = x_value_batch
    fetches = convert_keys_to_fetches(expected_keys)
    # (output, grad) = session.run([conv_layer, gradients_node],
    #                              feed_dict=call_model_args)
    # return {CONVOLUTION_GRADIENTS: grad, CONVOLUTION_LAYER: output}
    data = session.run(fetches, feed_dict=call_model_args)
    return {expected_key: data[i] for 
      (i, expected_key) in enumerate(expected_keys)}

  return call_model_function
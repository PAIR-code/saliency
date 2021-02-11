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

"""Utilities to run CallModelSaliency methods using TF1 models."""
from ..core.base import CONVOLUTION_LAYER_GRADIENTS
from ..core.base import CONVOLUTION_LAYER_VALUES
from ..core.base import OUTPUT_LAYER_GRADIENTS
import tensorflow.compat.v1 as tf

MISSING_Y_ERROR_MESSAGE = 'Cannot return key {} because no y was specified'
MISSING_CONV_LAYER_ERROR_MESSAGE = ('Cannot return key {} because no conv_layer'
                                    ' was specified')


def create_tf1_call_model_function(graph,
                                   session,
                                   y=None,
                                   x=None,
                                   conv_layer=None):
  """Constructs a SaliencyMask by computing dy/dx.

  Args:
    graph: The TensorFlow graph to evaluate masks on.
    session: The current TensorFlow session.
    y: The output tensor of the model. This tensor should be of size 1.
    x: The input tensor of the model. The outer dimension should be the batch
        size.
    conv_layer: The convolution layer tensor of the model. The outer
        dimension should be the batch size.

  Returns:
    call_model_function: A function that accepts the arguments used in the
        CallModelSaliency methods.
  """
  with graph.as_default():
    if x is None:
      raise ValueError('Expected input tensor for x but is equal to None.')
    if y is not None:
      output_gradients = tf.gradients(y, x)[0]
    if conv_layer is not None:
      conv_gradients = tf.gradients(conv_layer, x)[0]

  def convert_keys_to_fetches(expected_keys):
    fetches = []
    for expected_key in expected_keys:
      if expected_key == OUTPUT_LAYER_GRADIENTS:
        if y is None:
          raise RuntimeError(MISSING_Y_ERROR_MESSAGE.format(OUTPUT_LAYER_GRADIENTS))
        fetches.append(output_gradients)
      elif expected_key in [CONVOLUTION_LAYER_VALUES, CONVOLUTION_LAYER_GRADIENTS]:
        if conv_layer is None:
          raise RuntimeError(MISSING_CONV_LAYER_ERROR_MESSAGE.format(
              expected_key))
        if expected_key == CONVOLUTION_LAYER_VALUES:
          fetches.append(conv_layer)
        else:
          fetches.append(conv_gradients)
      else:
        raise ValueError('Invalid expected key {}'.format(expected_key))
    return fetches

  def call_model_function(x_value_batch,
                          call_model_args={},
                          expected_keys=None):
    # (output, grad) = session.run([conv_layer, gradients_node],
    #                              feed_dict=call_model_args)
    # return {CONVOLUTION_LAYER_GRADIENTS: grad, CONVOLUTION_LAYER_VALUES: output}
    with graph.as_default():
      fetches = convert_keys_to_fetches(expected_keys)
      call_model_args[x] = x_value_batch
      data = session.run(fetches, feed_dict=call_model_args)
      return {expected_key: data[i] for (i, expected_key) in
              enumerate(expected_keys)}

  return call_model_function

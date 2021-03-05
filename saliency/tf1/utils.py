# Copyright 2021 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities to run CallModelSaliency methods using TF1 models."""
from ..core.base import CONVOLUTION_LAYER_VALUES
from ..core.base import CONVOLUTION_OUTPUT_GRADIENTS
from ..core.base import INPUT_OUTPUT_GRADIENTS
from ..core.base import OUTPUT_LAYER_VALUES
tf = None

def _import_tf():
  """ Tries to import tensorflow.
  """
  global tf
  if tf is None:
    import tensorflow as tf
  return tf

MISSING_Y_ERROR_MESSAGE = 'Cannot return key {} because no y was specified'
MISSING_CONV_LAYER_ERROR_MESSAGE = ('Cannot return key {} because no conv_layer'
                                    ' was specified')


def create_tf1_call_model_function(graph,
                                   session,
                                   y,
                                   x,
                                   conv_layer=None):
  """Creates a call_model_function which calls the TF1 model specified.

  Args:
    graph: The TensorFlow graph to evaluate masks on.
    session: The current TensorFlow session.
    y: The output tensor of the model. This tensor shape should be (None,), 
      (batch_size), or () if not batching inputs.
    x: The input tensor of the model. The outer dimension should be the batch
      size.
    conv_layer: (Optional) The convolution layer tensor of the model. The outer
      dimension should be the batch size.

  Raises:
    ValueError: If input tensor x is missing.

  Returns:
    call_model_function: A function with the following signature:
        call_model_function(x_value_batch,
                              call_model_args=None,
                              expected_keys=None):
          x_value_batch - Input for the model, given as a batch (i.e. dimension
            0 is the batch dimension, dimensions 1 through n represent a single
            input).
          call_model_args - Other arguments used to call and run the model.
          expected_keys - List of keys that are expected in the output.
  """
  _import_tf()

  with graph.as_default():
    input_gradients = tf.compat.v1.gradients(y, x)[0]
    if conv_layer is not None:
      conv_gradients = tf.compat.v1.gradients(y, conv_layer)[0]

  def convert_keys_to_fetches(expected_keys):
    """Converts expected keys into an array of fetchable tensors.

    Args:
      expected_keys: Array of strings, representing the expected keys.

    Raises:
        RuntimeError: If tensor required for expected_key doesn't exist.

    Returns:
      Array of fetches that can be used in a session.run call.
    """
    fetches = []
    for expected_key in expected_keys:
      if expected_key==CONVOLUTION_LAYER_VALUES:
        if conv_layer is None:
          raise RuntimeError(MISSING_CONV_LAYER_ERROR_MESSAGE.format(
              expected_key))
        else:
          fetches.append(conv_layer)
      elif expected_key==CONVOLUTION_OUTPUT_GRADIENTS:
        if conv_layer is None:
          raise RuntimeError(MISSING_CONV_LAYER_ERROR_MESSAGE.format(
              expected_key))
        else:
          fetches.append(conv_gradients)
      elif expected_key==INPUT_OUTPUT_GRADIENTS:
        fetches.append(input_gradients)
      elif expected_key==OUTPUT_LAYER_VALUES:
        fetches.append(y)
      else:
        raise ValueError('Invalid expected key {}'.format(expected_key))
    return fetches

  def call_model_function(x_value_batch,
                          call_model_args=None,
                          expected_keys=None):
    """Calls a TF1 model and returns the expected keys.

    Args:
      x_value_batch: Input for the model, given as a batch (i.e. dimension
        0 is the batch dimension, dimensions 1 through n represent a single
        input).
      call_model_args: Other arguments used to call and run the model. Default
        is an empty dictionary.
      expected_keys: List of keys that are expected in the output.

    Raises:
        RuntimeError: If tensor required for expected_key doesn't exist.

    Returns:
      A dictionary of values corresponding to the output when running the model
        with x_value batch.
    """
    if call_model_args is None:
      call_model_args = {}
    with graph.as_default():
      fetches = convert_keys_to_fetches(expected_keys)
      call_model_args[x] = x_value_batch
      data = session.run(fetches, feed_dict=call_model_args)
      return {expected_key: data[i] for (i, expected_key) in
              enumerate(expected_keys)}

  return call_model_function

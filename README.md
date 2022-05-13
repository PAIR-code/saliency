# Saliency Library
## Updates

&#x1F534;&nbsp;&nbsp; Now framework-agnostic! [(Example core notebook)](Examples_core.ipynb) &nbsp;&#x1F534;

&#x1F517;&nbsp;&nbsp; For further explanation of the methods and more examples of the resulting maps, see our [Github Pages website](https://pair-code.github.io/saliency)  &nbsp;&#x1F517;

If upgrading from an older version, update old imports to `import saliency.tf1 as saliency`. We provide wrappers to make the framework-agnostic version compatible with TF1 models. [(Example TF1 notebook)](Examples_tf1.ipynb)

&#x1F534;&nbsp;&nbsp; Added Performance Information Curve (PIC) - a human
independent metric for evaluating the quality of saliency methods.
([Example notebook](https://github.com/PAIR-code/saliency/blob/master/pic_metrics.ipynb)) &nbsp;&#x1F534;

## Saliency Methods

This repository contains code for the following saliency techniques:

*   Guided Integrated Gradients* ([paper](https://arxiv.org/abs/2106.09788), [poster](https://github.com/PAIR-code/saliency/blob/master/docs/CVPR_Guided_IG_Poster.pdf))
*   XRAI* ([paper](https://arxiv.org/abs/1906.02825), [poster](https://github.com/PAIR-code/saliency/blob/master/docs/ICCV_XRAI_Poster.pdf))
*   SmoothGrad* ([paper](https://arxiv.org/abs/1706.03825))
*   Vanilla Gradients
    ([paper](https://scholar.google.com/scholar?q=Visualizing+higher-layer+features+of+a+deep+network&btnG=&hl=en&as_sdt=0%2C22),
    [paper](https://arxiv.org/abs/1312.6034))
*   Guided Backpropogation ([paper](https://arxiv.org/abs/1412.6806))
*   Integrated Gradients ([paper](https://arxiv.org/abs/1703.01365))
*   Occlusion
*   Grad-CAM ([paper](https://arxiv.org/abs/1610.02391))
*   Blur IG ([paper](https://arxiv.org/abs/2004.03383))

\*Developed by PAIR.

This list is by no means comprehensive. We are accepting pull requests to add
new methods!

## Evaluation of Saliency Methods

The repository provides an implementation of Performance Information Curve (PIC) -
a human independent metric for evaluating the quality of saliency methods
([paper](https://arxiv.org/abs/1906.02825),
[poster](https://github.com/PAIR-code/saliency/blob/master/docs/ICCV_XRAI_Poster.pdf),
[code](https://github.com/PAIR-code/saliency/blob/master/saliency/metrics/pic.py),
[notebook](https://github.com/PAIR-code/saliency/blob/master/pic_metrics.ipynb)).


## Download

```
# To install the core subpackage:
pip install saliency

# To install core and tf1 subpackages:
pip install saliency[tf1]

```

or for the development version:
```
git clone https://github.com/pair-code/saliency
cd saliency
```


## Usage

The saliency library has two subpackages:
*	`core` uses a generic `call_model_function` which can be used with any ML 
	framework.
*	`tf1` accepts input/output tensors directly, and sets up the necessary 
	graph operations for each method.

### Core

Each saliency mask class extends from the `CoreSaliency` base class. This class
contains the following methods:

*   `GetMask(x_value, call_model_function, call_model_args=None)`: Returns a mask
    of
    the shape of non-batched `x_value` given by the saliency technique.
*   `GetSmoothedMask(x_value, call_model_function, call_model_args=None, stdev_spread=.15, nsamples=25, magnitude=True)`: 
    Returns a mask smoothed of the shape of non-batched `x_value` with the 
    SmoothGrad technique.


The visualization module contains two methods for saliency visualization:

* ```VisualizeImageGrayscale(image_3d, percentile)```: Marginalizes across the
  absolute value of each channel to create a 2D single channel image, and clips
  the image at the given percentile of the distribution. This method returns a
  2D tensor normalized between 0 to 1.
* ```VisualizeImageDiverging(image_3d, percentile)```: Marginalizes across the
  value of each channel to create a 2D single channel image, and clips the
  image at the given percentile of the distribution. This method returns a
  2D tensor normalized between -1 to 1 where zero remains unchanged.

If the sign of the value given by the saliency mask is not important, then use
```VisualizeImageGrayscale```, otherwise use ```VisualizeImageDiverging```. See
the SmoothGrad paper for more details on which visualization method to use.

##### call_model_function
`call_model_function` is how we pass inputs to a given model and receive the outputs
necessary to compute saliency masks. The description of this method and expected 
output format is in the `CoreSaliency` description, as well as separately for each method.


##### Examples

[This example iPython notebook](http://github.com/pair-code/saliency/blob/master/Examples_core.ipynb)
showing these techniques is a good starting place.

Here is a condensed example of using IG+SmoothGrad with TensorFlow 2:

```
import saliency.core as saliency
import tensorflow as tf

...

# call_model_function construction here.
def call_model_function(x_value_batched, call_model_args, expected_keys):
	tape = tf.GradientTape()
	grads = np.array(tape.gradient(output_layer, images))
	return {saliency.INPUT_OUTPUT_GRADIENTS: grads}

...

# Load data.
image = GetImagePNG(...)

# Compute IG+SmoothGrad.
ig_saliency = saliency.IntegratedGradients()
smoothgrad_ig = ig_saliency.GetSmoothedMask(image, 
											call_model_function, 
                                            call_model_args=None)

# Compute a 2D tensor for visualization.
grayscale_visualization = saliency.VisualizeImageGrayscale(
    smoothgrad_ig)
```

### TF1

Each saliency mask class extends from the `TF1Saliency` base class. This class
contains the following methods:

*   `__init__(graph, session, y, x)`: Constructor of the SaliencyMask. This can
    modify the graph, or sometimes create a new graph. Often this will add nodes
    to the graph, so this shouldn't be called continuously. `y` is the output
    tensor to compute saliency masks with respect to, `x` is the input tensor
    with the outer most dimension being batch size.
*   `GetMask(x_value, feed_dict)`: Returns a mask of the shape of non-batched
    `x_value` given by the saliency technique.
*   `GetSmoothedMask(x_value, feed_dict)`: Returns a mask smoothed of the shape
    of non-batched `x_value` with the SmoothGrad technique.

The visualization module contains two visualization methods:

* ```VisualizeImageGrayscale(image_3d, percentile)```: Marginalizes across the
  absolute value of each channel to create a 2D single channel image, and clips
  the image at the given percentile of the distribution. This method returns a
  2D tensor normalized between 0 to 1.
* ```VisualizeImageDiverging(image_3d, percentile)```: Marginalizes across the
  value of each channel to create a 2D single channel image, and clips the
  image at the given percentile of the distribution. This method returns a
  2D tensor normalized between -1 to 1 where zero remains unchanged.

If the sign of the value given by the saliency mask is not important, then use
```VisualizeImageGrayscale```, otherwise use ```VisualizeImageDiverging```. See
the SmoothGrad paper for more details on which visualization method to use.

##### Examples

[This example iPython notebook](http://github.com/pair-code/saliency/blob/master/Examples_tf1.ipynb) shows
these techniques is a good starting place.

Another example of using GuidedBackprop with SmoothGrad from TensorFlow:

```
from saliency.tf1 import GuidedBackprop
from saliency.tf1 import VisualizeImageGrayscale
import tensorflow.compat.v1 as tf

...
# Tensorflow graph construction here.
y = logits[5]
x = tf.placeholder(...)
...

# Compute guided backprop.
# NOTE: This creates another graph that gets cached, try to avoid creating many
# of these.
guided_backprop_saliency = GuidedBackprop(graph, session, y, x)

...
# Load data.
image = GetImagePNG(...)
...

smoothgrad_guided_backprop =
    guided_backprop_saliency.GetMask(image, feed_dict={...})

# Compute a 2D tensor for visualization.
grayscale_visualization = visualization.VisualizeImageGrayscale(
    smoothgrad_guided_backprop)
```

## Conclusion/Disclaimer

If you have any questions or suggestions for improvements to this library,
please contact the owners of the `PAIR-code/saliency` repository.

This is not an official Google product.
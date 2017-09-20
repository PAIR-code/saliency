# Saliency Methods

## Introduction

This repository contains code for [SmoothGrad](https://pair-code.github.io/saliency/), as well as implementations of
several other saliency techniques. Each of these techniques can also be
augmented with SmoothGrad. The techniques implemented in this library are:

*   Vanilla Gradients
    ([paper](https://scholar.google.com/scholar?q=Visualizing+higher-layer+features+of+a+deep+network&btnG=&hl=en&as_sdt=0%2C22),
    [paper](https://arxiv.org/abs/1312.6034))
*   Guided Backpropogation ([paper](https://arxiv.org/abs/1412.6806))
*   Integrated Gradients ([paper](https://arxiv.org/abs/1703.01365))
*   Occlusion
*   Grad-CAM ([paper](https://arxiv.org/abs/1610.02391))

This list is by no means comprehensive. We are accepting pull requests to add
new methods!

## Download
```
git clone https://github.com/pair-code/saliency
cd saliency
```

## Usage

Each saliency mask class extends from the `SaliencyMask` base class. This class
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

## Examples

[This example iPython notebook](http://github.com/pair-code/saliency/blob/master/Examples.ipynb) shows
these techniques is a good starting place.

Another example of using GuidedBackprop with SmoothGrad from TensorFlow:

```
from guided_backprop import GuidedBackprop
import visualization

...
# Tensorflow graph construction here.
y = logits[5]
x = tf.placeholder(...)
...

# Compute guided backprop.
# NOTE: This creates another graph that gets cached, try to avoid creating many
# of these.
guided_backprop_saliency = GuidedBackpropSaliency(graph, session, y, x)

...
# Load data.
image = GetImagePNG(...)
...

smoothgrad_guided_backprop =
    guided_backprop_saliency.GetSmoothedMask(image, feed_dict={...})

# Compute a 2D tensor for visualization.
grayscale_visualization = visualization.VisualizeImageGrayscale(
    smoothgrad_guided_backprop)
```

This is not an official Google product.

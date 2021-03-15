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

from ..core import visualization


VisualizeImageGrayscale = visualization.VisualizeImageGrayscale
# def VisualizeImageGrayscale(image_3d, percentile=99):
#   r"""Returns a 3D tensor as a grayscale 2D tensor.

#   This method sums a 3D tensor across the absolute value of axis=2, and then
#   clips values at a given percentile.
#   """

VisualizeImageDiverging = visualization.VisualizeImageDiverging
# def VisualizeImageDiverging(image_3d, percentile=99):
#   r"""Returns a 3D tensor as a 2D tensor with positive and negative values.
#   """
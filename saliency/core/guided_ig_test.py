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

"""Tests the core implementation of Guided IG."""
import unittest

from . import guided_ig
from .base import INPUT_OUTPUT_GRADIENTS
import numpy as np


class GuidedIGTest(unittest.TestCase):
  """To run: "python -m unittest saliency.core.guided_ig_test" from the top-level directory."""

  def _linear_test_func(self, x_value_batch, call_model_args, expected_keys):
    """Returns gradient of a linear function 2 * x1 + x2.

    Guided IG for linear function produces tha same attribution as Integrated
    Gradients regardless of the parameters.
    """
    self.assertListEqual(expected_keys, [INPUT_OUTPUT_GRADIENTS])
    self.assertEqual(call_model_args, 'airplane')
    return {INPUT_OUTPUT_GRADIENTS: [[2.0, 1.0]]}

  def _quadratic_indep_test_func(self, x_value_batch, call_model_args,
      expected_keys):
    """Returns gradient of a quadratic function 2 * x1^2 + x2^2.

    The partial derivatives of the function with respect to x1 and x2 are
    independent from each other. For such functions, the Guided IG result
    should be equivalent to regular Integrated Gradients.
    """
    self.assertListEqual(expected_keys, [INPUT_OUTPUT_GRADIENTS])
    return {INPUT_OUTPUT_GRADIENTS: [
        [4 * x_value_batch[0, 0], 2 * x_value_batch[0, 1]]]}

  def _quadratic_dep_test_func(self, x_value_batch, call_model_args,
      expected_keys):
    """Returns gradient of a quadratic function (2 * x1 + x2)^2.

    The partial derivatives of the function with respect to x1 and x2 depend on
    each other. For such functions, the Guided IG produces different result
    than Integrated Gradients. TYhe difference depends on the max_dist that
    Guided IG can deviate from the IG path.
    """
    self.assertListEqual(expected_keys, [INPUT_OUTPUT_GRADIENTS])
    return {INPUT_OUTPUT_GRADIENTS: [
        [8 * x_value_batch[0, 0] + 4 * x_value_batch[0, 1],
         4 * x_value_batch[0, 0] + 2 * x_value_batch[0, 1]]]}

  def testGuidedIGLinearWithBaseline(self):
    """Tests attributions of the linear function with provided baseline.

    The expected result should be equal to the regular Integrated Gradients
    attribution.
    """
    gig = guided_ig.GuidedIG()
    mask = gig.GetMask(x_value=[10.0, 5.0],
                       call_model_function=self._linear_test_func,
                       call_model_args='airplane',
                       x_baseline=[-1.0, -3.0],
                       x_steps=10,
                       fraction=0.1,
                       max_dist=0.5)
    np.testing.assert_allclose(mask, [22.0, 8.0])

  def testLinearWithoutBaseline(self):
    """Tests attributions of the linear function with default baseline.

    The expected result should be equal to the regular Integrated Gradients
    attribution.
    """
    gig = guided_ig.GuidedIG()
    mask = gig.GetMask(x_value=[10, 5],
                       call_model_function=self._linear_test_func,
                       call_model_args='airplane',
                       x_steps=10,
                       fraction=0.1,
                       max_dist=0.5)
    np.testing.assert_allclose(mask, [20.0, 5.0])

  def testIndependentQuadratic(self):
    """Tests attributions of the quadratic function with independent partials.

    The expected result should be equal to the regular Integrated Gradients
    attribution regardless of the parameters.
    """
    gig = guided_ig.GuidedIG()
    mask = gig.GetMask(x_value=[3, 3],
                       call_model_function=self._quadratic_indep_test_func,
                       x_steps=1000,
                       fraction=0.1,
                       max_dist=0.4)
    np.testing.assert_allclose(mask, [18.0, 9.0], rtol=0.01)

  def testIndependentQuadraticInputIsLessThanBaseline(self):
    """Tests attributions of the quadratic function with independent partials.

    The expected result should be equal to the regular Integrated Gradients
    attribution regardless of the parameters. This method tests the case when
    the baseline values are higher than the input.
    """
    gig = guided_ig.GuidedIG()
    mask = gig.GetMask(x_value=[0, 0],
                       call_model_function=self._quadratic_indep_test_func,
                       x_baseline=[3, 3],
                       x_steps=1000,
                       fraction=0.1,
                       max_dist=0.4)
    np.testing.assert_allclose(mask, [-18.0, -9.0], rtol=0.01)

  def testDependentQuadraticIG(self):
    """Tests attributions of the quadratic function with dependent partials.

    The test sets such parameters that produce attribution equal to
    Integrated Gradients, namely, by setting zero deviation from the
    straight-line path or changing all features at every integration step.
    """
    gig = guided_ig.GuidedIG()
    # Setting max_dist to 0.0 reduces Guided IG to regular IG.
    mask = gig.GetMask(x_value=[3, 3],
                       call_model_function=self._quadratic_dep_test_func,
                       x_steps=1000,
                       fraction=0.1,
                       max_dist=0.0)
    np.testing.assert_allclose(mask, [54.0, 27.0], rtol=0.01)

    # Setting fraction to 1.0 also reduces Guided IG to regular IG.
    mask = gig.GetMask(x_value=[3, 3],
                       call_model_function=self._quadratic_dep_test_func,
                       x_steps=1000,
                       fraction=1.0,
                       max_dist=0.5)
    np.testing.assert_allclose(mask, [54.0, 27.0], rtol=0.01)

  def testDependentQuadraticUnbound(self):
    """Tests unbounded Guided IG with quadratic function and dependent partials.
    """
    gig = guided_ig.GuidedIG()
    mask = gig.GetMask(x_value=[3, 3],
                       call_model_function=self._quadratic_dep_test_func,
                       x_steps=1000,
                       fraction=0.1,
                       max_dist=1.0)
    np.testing.assert_allclose(mask, [72.0, 9.0], rtol=0.01)

  def testDependentQuadratic(self):
    """Tests bounded Guided IG with quadratic function and dependent partials.

    The expected result should be between attributions of Integrated Gradients
    and unbounded IG. The test checks that the result attribution satisfies the
    completeness axiom.
    """
    gig = guided_ig.GuidedIG()
    mask = gig.GetMask(x_value=[3, 3],
                       call_model_function=self._quadratic_dep_test_func,
                       x_steps=1000,
                       fraction=0.1,
                       max_dist=0.2)
    # Check the result satisfy completeness.
    self.assertAlmostEqual(mask.sum(), 81.0, delta=0.1)
    # Check that the result is between unbounded Guided IG and IG.
    self.assertTrue(55.0 < mask[0] < 71.0)
    self.assertTrue(10.0 < mask[1] < 26.0)

if __name__ == '__main__':
  unittest.main()

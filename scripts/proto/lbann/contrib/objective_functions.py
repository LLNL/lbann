"""Experimental objective functions.

These are mostly unsupported.

"""

import lbann.proto as lp
import lbann.modules as lm
from collections.abc import Iterable

class CrossEntropyWithUncertainty(lm.Module):
    def forward(self, inputs):
        if len(inputs) != 2:
            raise ValueError('expected two inputs: predictions and labels')
        pred = inputs[0]
        label = inputs[1]   # Assumed to be Boolean
        masked_pred = lp.Multiply([pred, label])
        pred_sum = lp.Reduction(masked_pred)
        return lp.Negative(lp.Log(pred_sum))

class GeometricDistributionNegativeLogLikelihood(lm.Module):
    def forward(self, inputs):
        if len(inputs) != 2:
            raise ValueError('expected two inputs: predictions and labels')
        pred = inputs[0]
        label = inputs[1]
        ones = p.Constant(hint_layer=pred, value=1.0)
        term1 = lp.Multiply([label, lp.Log(lp.Subtract([ones, pred]))])
        term2 = lp.Log(pred)
        full = lp.WeightedSum([term1, term2], scaling_factors='-1.0 -1.0')
        return lp.Reduction(full)

class PoissonDistributionNegativeLogLikelihood(lm.Module):
    def forward(self, inputs):
        raise NotImplementedError   # Requires log-gamma function
        if len(inputs) != 2:
            raise ValueError('expected two inputs: predictions and labels')
        pred = inputs[0]
        label = inputs[1]
        ones = lp.Constant(hint_layer=pred, value=1.0)
        term1 = pred
        term2 = lp.Multiply([label, lp.Log(pred)])
        term3 = lp.LogGamma(lp.Add([label, ones]))
        full = lp.WeightedSum([term1, term2, term3], scaling_factors='1.0 -1.0 1.0')
        return lp.Reduction(full)

class PolyaDistributionNegativeLogLikelihood(lm.Module):
    def forward(self, inputs):
        raise NotImplementedError   # Requires log-gamma function
        if len(inputs) != 2:
            raise ValueError('expected two inputs: predictions and labels')
        pred = inputs[0]
        label = inputs[1]
        count = lp.Reduction(label)
        alpha_sum = lp.Reduction(pred)
        lgamma_alpha_sum = lp.Reduction(lp.LogGamma(pred))
        lgamma_alpha_level_count_sum = lp.Reduction(lp.LogGamma(lp.Add([pred, label])))
        return lp.WeightedSum([lp.LogGamma(alpha_sum),
                               lp.LogGamma(lp.Sum([count, alpha_sum])),
                               lgamma_alpha_level_count,
                               lgamma_alpha_sum],
                              scaling_factors='-1.0 1.0 -1.0 1.0')

class GroupLasso(lm.Module):
    def __init__(self, weights, height, width):
        self.weights = weights
        self.height = height
        self.width = width
    def forward(self, _):
        w = lp.WeightsLayer(weights=self.weights, dims='%d %d'.format(self.width, self.height))
        slice = lp.Slice(w, axis=0, slice_points=' '.join(range(self.width+1)))
        cols = []
        for _ in range(self.width):
            cols.append(lp.Sqrt(lp.L2Norm2(slice)))
        return lp.Sum(cols)

class L1WeightRegularization(lm.Module):
    def __init__(self, weights, dims):
        self.weights = weights
        self.dims = dims
    def forward(self, _):
        w = lp.WeightsLayer(weights=self.weights, dims=' '.join(self.dims))
        return lp.L1Norm(w)

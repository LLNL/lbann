"""Experimental objective functions.

These are mostly unsupported.

"""

from collections.abc import Iterable
import lbann
import lbann.modules

class CrossEntropyWithUncertainty(lbann.modules.Module):
    def forward(self, inputs):
        if len(inputs) != 2:
            raise ValueError('expected two inputs: predictions and labels')
        pred = inputs[0]
        label = inputs[1]   # Assumed to be Boolean
        masked_pred = lbann.Multiply([pred, label])
        pred_sum = lbann.Reduction(masked_pred)
        return lbann.Negative(lbann.Log(pred_sum))

class GeometricDistributionNegativeLogLikelihood(lbann.modules.Module):
    def forward(self, inputs):
        if len(inputs) != 2:
            raise ValueError('expected two inputs: predictions and labels')
        pred = inputs[0]
        label = inputs[1]
        ones = p.Constant(hint_layer=pred, value=1.0)
        term1 = lbann.Multiply([label, lbann.Log(lbann.Subtract([ones, pred]))])
        term2 = lbann.Log(pred)
        full = lbann.WeightedSum([term1, term2], scaling_factors='-1.0 -1.0')
        return lbann.Reduction(full)

class PoissonDistributionNegativeLogLikelihood(lbann.modules.Module):
    def forward(self, inputs):
        raise NotImplementedError   # Requires log-gamma function
        if len(inputs) != 2:
            raise ValueError('expected two inputs: predictions and labels')
        pred = inputs[0]
        label = inputs[1]
        ones = lbann.Constant(hint_layer=pred, value=1.0)
        term1 = pred
        term2 = lbann.Multiply([label, lbann.Log(pred)])
        term3 = lbann.LogGamma(lbann.Add([label, ones]))
        full = lbann.WeightedSum([term1, term2, term3], scaling_factors='1.0 -1.0 1.0')
        return lbann.Reduction(full)

class PolyaDistributionNegativeLogLikelihood(lbann.modules.Module):
    def forward(self, inputs):
        raise NotImplementedError   # Requires log-gamma function
        if len(inputs) != 2:
            raise ValueError('expected two inputs: predictions and labels')
        pred = inputs[0]
        label = inputs[1]
        count = lbann.Reduction(label)
        alpha_sum = lbann.Reduction(pred)
        lgamma_alpha_sum = lbann.Reduction(lbann.LogGamma(pred))
        lgamma_alpha_level_count_sum = lbann.Reduction(lbann.LogGamma(lbann.Add([pred, label])))
        return lbann.WeightedSum([lbann.LogGamma(alpha_sum),
                                  lbann.LogGamma(lbann.Sum([count, alpha_sum])),
                                  lgamma_alpha_level_count,
                                  lgamma_alpha_sum],
                                 scaling_factors='-1.0 1.0 -1.0 1.0')

class GroupLasso(lbann.modules.Module):
    def __init__(self, weights, height, width):
        self.weights = weights
        self.height = height
        self.width = width
    def forward(self, _):
        w = lbann.WeightsLayer(weights=self.weights, dims='%d %d'.format(self.width, self.height))
        slice = lbann.Slice(w, axis=0, slice_points=' '.join(range(self.width+1)))
        cols = []
        for _ in range(self.width):
            cols.append(lbann.Sqrt(lbann.L2Norm2(slice)))
        return lbann.Sum(cols)

class L1WeightRegularization(lbann.modules.Module):
    def __init__(self, weights, dims):
        self.weights = weights
        self.dims = dims
    def forward(self, _):
        w = lbann.WeightsLayer(weights=self.weights, dims=' '.join(self.dims))
        return lbann.L1Norm(w)

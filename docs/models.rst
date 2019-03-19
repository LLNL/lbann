Models
=================================

A model is a collection of layers that are composed into a
computational graph.  The model also holds the weight matrices for
each learning layer.  During training the weight matrices are the free
parameters.  For a trained network during inference the weight matrics
are preloaded from saved matrices.  The model also contains the
objective function and optimizer classes for the weights.

.. autodoxygenindex::
 :project: models

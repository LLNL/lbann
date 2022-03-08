.. role:: python(code)
          :language: python


.. _loss-layers:

====================================
Loss Layers
====================================

.. csv-table::
   :header: "Layer", "Description"
   :widths: auto

   :ref:`CategoricalAccuracy`, "0-1 loss function"
   :ref:`CrossEntropy`, "Cross entropy between probability vectors"
   :ref:`L1Norm`, "L1 vector norm"
   :ref:`L2Norm2`, "Square of L2 vector norm"
   :ref:`MeanAbsoluteError`, "Mean absolute error"
   :ref:`MeanSquaredError`, "Mean squared error"
   :ref:`TopKCategoricalAccuracy`, "Top-k prediction scores"

________________________________________


.. _CategoricalAccuracy:

----------------------------------------
CategoricalAccuracy
----------------------------------------

The Categorical Accuracy Layer is a 0-1 loss function.

Requires two inputs, which are respectively interpreted as prediction
scores and as a one-hot label vector. The output is one if the top
entries in both inputs are in the same position and is otherwise
zero. Ties are broken in favor of entries with smaller indices.

This is primarily intended for use as a metric since it is not
differentiable.

Arguments: None

:ref:`Back to Top<loss-layers>`

________________________________________


.. _CrossEntropy:

----------------------------------------
CrossEntropy
----------------------------------------

Cross entropy between probability vectors.

Given a predicted distribution :math:`y` and ground truth distribution
:math:`\hat{y}`,

.. math::

   CE(y,\hat{y}) = - \sum\limits_{i} \hat{y}_i \log y_i

Arguments:

   :use_labels: (``bool``) Advanced option for distconv

:ref:`Back to Top<loss-layers>`

________________________________________


.. _L1Norm:

----------------------------------------
L1Norm
----------------------------------------

L1 vector norm

.. math::

   \lVert x\rVert_1 = \sum\limits_{i} | x_i |

Arguments: None

:ref:`Back to Top<loss-layers>`

________________________________________


.. _L2Norm2:

----------------------------------------
L2Norm2
----------------------------------------

Square of L2 vector norm

.. math::

   \lVert x\rVert_2^2 = \sum\limits_{i} x_i^2

Arguments: None

:ref:`Back to Top<loss-layers>`

________________________________________


.. _MeanAbsoluteError:

----------------------------------------
MeanAbsoluteError
----------------------------------------

Given a prediction :math:`y` and ground truth :math:`\hat{y}`,

.. math::

   MAE(y,\hat{y})
   = \frac{1}{n} \sum\limits_{i=1}^{n} | y_i - \hat{y}_i |

Arguments: None

:ref:`Back to Top<loss-layers>`

________________________________________


.. _MeanSquaredError:

----------------------------------------
MeanSquaredError
----------------------------------------

Given a prediction :math:`y` and ground truth :math:`\hat{y}`,

.. math::

   MSE(y,\hat{y})
   = \frac{1}{n} \sum\limits_{i=1}^{n} (y_i - \hat{y}_i)^2

Arguments: None

:ref:`Back to Top<loss-layers>`

________________________________________


.. _TopKCategoricalAccuracy:

----------------------------------------
TopKCategoricalAccuracy
----------------------------------------

Requires two inputs, which are respectively interpreted as prediction
scores and as a one-hot label vector. The output is one if the
corresponding label matches one of the top-k prediction scores and is
otherwise zero. Ties in the top-k prediction scores are broken in
favor of entries with smaller indices.

Arguments:

   :k: (``int64``)

:ref:`Back to Top<loss-layers>`

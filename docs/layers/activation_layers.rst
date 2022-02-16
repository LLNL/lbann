.. role:: python(code)
          :language: python


.. _activation-layers:

====================================
Activation Layers
====================================

.. csv-table::
   :header: "Layer", "Description"
   :widths: auto

   :ref:`Elu`, "Exponential linear unit"
   :ref:`Identity`, "Output the input tensor"
   :ref:`LeakyRelu`, "Leaky relu"
   :ref:`LogSoftmax`, "Logarithm of softmax function"
   :ref:`Relu`, "Rectified linear unit"
   :ref:`Softmax`, "Softmax"

________________________________________


.. _Elu:

----------------------------------------
Elu
----------------------------------------

Exponential linear unit

.. math::

   \text{ELU}(x; \alpha) =
   \begin{cases}
      x                & x > 0 \\
      \alpha (e^x - 1) & x \leq 0
   \end{cases}

:math:`alpha` should be non-negative. See:

Djork-Arne Clevert, Thomas Unterthiner, and Sepp Hochreiter. "Fast and
accurate deep network learning by exponential linear units (ELUs)."
arXiv preprint arXiv:1511.07289 (2015).

Arguments:

   :alpha: (``double``, optional) Default = 1. Should be >=0

:ref:`Back to Top<activation-layers>`

________________________________________


.. _Identity:

----------------------------------------
Identity
----------------------------------------

Output the input tensor

This layer is very cheap since it just involves setting up tensor
views.

Arguments: None

:ref:`Back to Top<activation-layers>`

________________________________________


.. _LeakyRelu:

----------------------------------------
LeakyRelu
----------------------------------------

.. math::

   \text{LeakyReLU}(x; \alpha) =
      \begin{cases}
         x        & x > 0 \\
         \alpha x & x \leq 0
    \end{cases}

See:

Andrew L. Maas, Awni Y. Hannun, and Andrew Y. Ng. "Rectifier
nonlinearities improve neural network acoustic models." In Proc. ICML,
vol. 30, no. 1, p. 3. 2013.

Arguments:

   :negative_slope: (``double``, optional) Default = 0.01

:ref:`Back to Top<activation-layers>`

________________________________________


.. _LogSoftmax:

----------------------------------------
LogSoftmax
----------------------------------------

Logarithm of softmax function

:math:`\log \text{softmax}(x)_i = x_i - \log \sum_j e^{x_j}`

Arguments: None

:ref:`Back to Top<activation-layers>`

________________________________________


.. _Relu:

----------------------------------------
Relu
----------------------------------------

Rectified linear unit

:math:`ReLU(x) = \text{max}(x, 0)`

Arguments: None

:ref:`Back to Top<activation-layers>`

________________________________________


.. _Softmax:

----------------------------------------
Softmax
----------------------------------------

:math:`\text{softmax}(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}}`

Arguments:

   :softmax_mode: (``string``, optional) instance (default), channel

:ref:`Back to Top<activation-layers>`

________________________________________

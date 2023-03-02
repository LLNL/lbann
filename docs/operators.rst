.. role:: python(code)
          :language: python

.. _operators:

============================================================
Operators
============================================================

An operator defines a mathematical function that that supports both
forward and possibly backward operations. In the forward direction, it
takes a vector of input tensors and produces a vector of output
tensors.  In the backward direction they implement the differentiation
of the forward operation, applying the function to the operator's
forward inputs and gradient with respect to the outputs, to compute
the gradient with respect to the input.

Operators act as curried functions, they can have state that is
defined during construction but do not hold internal state.  A
operator should also be able to take objective function gradients
w.r.t. the outputs ("previous error signals") and compute the
objective function gradients w.r.t. the inputs ("error signals"). This
allows the model to perform automatic differentiation.

Operators are specified for unique input and output data types.


.. _abs:

------------------------------------------------
Abs
------------------------------------------------

Perform entrywise absolute value on input tensor.

.. math::

  \text{abs}(x) = |x|


.. _add:

------------------------------------------------
Add
------------------------------------------------

Perform entrywise addition on two input tensors.


.. _add-constant:

------------------------------------------------
Add Constant
------------------------------------------------

Add a constant to each input value.


.. _clamp:

------------------------------------------------
Clamp
------------------------------------------------

Constrain values to a range.

.. math::

   \text{clamp}(x; \text{min}, \text{max}) =
       \begin{cases}
         \text{min} & x \leq \text{min}           \\
         x          & \text{min} < x < \text{max} \\
         \text{max} & x \geq \text{max}
       \end{cases}


.. _constant-subtract:

------------------------------------------------
Constant Subtract
------------------------------------------------

Subtract each input value from a constant.

.. math::

   x = (c - x)


.. _cosine:

------------------------------------------------
Cos
------------------------------------------------

Calculate entrywise cosine of the input tensor.


.. _equal-constant:

------------------------------------------------
Equal Constant
------------------------------------------------

Perform entrywise logical equal on input tensor and a constant.


.. _multiply:

------------------------------------------------
Multiply
------------------------------------------------

Perform entrywise multiplication on input tensors.


.. _not-equal-constant:

------------------------------------------------
Not Equal Constant
------------------------------------------------

Perform entrywise logical not equal on input tensor and a constant.


.. _scale:

------------------------------------------------
Scale
------------------------------------------------

Scale each input value by a constant.

.. math::

   x = c * x


.. _sin:

------------------------------------------------
Sin
------------------------------------------------

Calculate entrywise sin of the input tensor.


.. _subtract:

------------------------------------------------
Subtract
------------------------------------------------

Perform entrywise subtraction on two input tensors.

.. math::

   \text{subtract}(x, y) \\
   z = x - y


.. _subtract-constant:

------------------------------------------------
Subtract Constant
------------------------------------------------

Subtract a constant from each input value.

.. math::

   x = (x - c)

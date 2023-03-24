.. role:: python(code)
          :language: python

.. _operator-layer:

====================================
Operator Layer
====================================

The :python:`OperatorLayer` is composed of one or more operator
objects (name chosen to avoid collision). Operators are applied
sequentially. For available operators see :ref:`operators`.

In the Python Front-End, there is an API short-hand for constructing
single-operator :python:`OperatorLayer` instances. For a given
operator, its name without the “Operator” suffix will create an
:python:`OperatorLayer` with that operator as its held operator. For
example, :python:`lbann.Add(<layer-arguments>)` will create an
:python:`lbann.OperatorLayer` equivalent to
:python:`lbann.OperatorLayer(<layer-arguments>, ops=[lbann.AddOperator()])`

Arguments:

   :ops: (``repeated Operator``)

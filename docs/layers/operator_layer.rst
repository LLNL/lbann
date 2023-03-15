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
single-operator `OperatorLayer` instances. For a given operator, its
name without the “Operator” suffix will create an `OperatorLayer` with
that operator as its held operator. For example,
`lbann.Add(<layer-arguments>)` will create an `lbann.OperatorLayer`
equivalent to `lbann.OperatorLayer(<layer-arguments>,
ops=[lbann.AddOperator()])`

Arguments:

   :ops: (``repeated Operator``)

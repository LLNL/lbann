.. role:: python(code)
          :language: python

.. _operator-layers:

====================================
Operator Layers
====================================

.. csv-table::
   :header: "Layer", "Description"
   :widths: auto

   :ref:`OperatorLayer`, "Layer composed of one or more operator
   objects (name chosen to avoid collision)"

.. csv-table::
   :header: "Operator", "Description"
   :widths: auto

   :ref:`Operator`, "Base class for LBANN operators"
   Clamp, "Constrain all values in a tensor within a range"
   Abs, "Apply the Abs operator entrywise"
   Acps, "Apply the Acos operator entrywise"
   Acosh, "Apply the Acosh operator entrywise"
   Add, "Apply the Add operator entrywise"
   AddConstant, "Add a constant to each input value (x+c)"
   Asin, "Apply the Asin operator entrywise"
   Asinh, "Apply the Asinh operator entrywise"
   Atan, "Apply the Atan operator entrywise"
   Atanh, "Apply the Atanh operator entrywise"
   BinaryCrossEntropy, "Apply the BinaryCrossEntropy operator entrywise"
   BooleanAccuracy, "Apply the BooleanAccuracy operator entrywise"
   BooleanFalseNegative, "Apply the BooleanFalseNegative operator
   entrywise"
   BooleanFalsePositive, "Apply the BooleanFalsePositive operator
   entrywise"
   Ceil, "Apply the Ceil operator entrywise"
   ConstantSubtract, "Subtract each input value from a constant (c-x)"
   Cos, "Apply the Cos operator entrywise"
   Cosh, "Apply the Cosh operator entrywise"
   Divide, "Apply the Divide operator entrywise"
   Equal, "Apply the Equal operator entrywise"
   EqualConstant, "Test each value for equality with a constant
   (x==c)"
   Erf, "Apply the Erf operator entrywise"
   ErfInv, "Apply the ErfInv operator entrywise"
   Exp, "Apply the Exp operator entrywise"
   Expm1, "Apply the Expm1 operator entrywise"
   Floor, "Apply the Floor operator entrywise"
   Greater, "Apply the Greater operator entrywise"
   GreaterConstant, "Test each value for 'greater-than' with a
   constant (x>c)"
   GreaterEqual, "Apply the GreaterEqual operator entrywise"
   GreaterEqualConstant, "Test each value for
   'greater-than-or-equal-to' with a constant (x>=c)"
   Less, "Apply the Less operator entrywise"
   LessConstant, "Test each value for 'less-than' with a constant
   (x<c)"
   LessEqual, "Apply the LessEqual operator entrywise"
   LessEqualConstant, "Test each value for 'less-than-or-equal-to' with
   a constant (x<=c)"
   Log, "Apply the Log operator entrywise"
   Log1p, "Apply the Log1p operator entrywise"
   LogSigmoid, "Apply the LogSigmoid operator entrywise"
   LogicalAnd, "Apply the LogicalAnd operator entrywise"
   LogicalNot, "Apply the LogicalNot operator entrywise"
   LogicalOr, "Apply the LogicalOr operator entrywise"
   LogicalXor, "Apply the LogicalXor operator entrywise"
   Max, "Apply the Max operator entrywise"
   Min, "Apply the Min operator entrywise"
   Mod, "Apply the Mod operator entrywise"
   Multiply, "Apply the Multiply operator entrywise"
   Negative, "Apply the Log Negative entrywise"
   NotEqual, "Apply the NotEqual operator entrywise"
   NotEqualConstant, "Test each value for inequality with a constant
   (x!=c)"
   Pow, "Apply the Pow operator entrywise"
   Reciprocal, "Apply the Reciprocal operator entrywise"
   Round, "Apply the Round operator entrywise"
   Rsqrt, "Apply the Rsqrt operator entrywise"
   SafeDivide, "Apply the SafeDivide operator entrywise"
   SafeReciprocal, "Apply the SafeReciprocal operator entrywise"
   Scale, "Scale each input value by a constant value (c*x)"
   Selu, "Apply the Selu operator entrywise"
   Sigmoid, "Apply the Sigmoid operator entrywise"
   SigmoidBinaryCrossEntropy, "Apply the SigmoidBinaryCrossEntropy
   operator entrywise."
   Sign, "Apply the Sign operator entrywise"
   Sin, "Apply the Sin operator entrywise"
   Sinh, "Apply the Sinh operator entrywise"
   Softplus, "Apply the Softplus operator entrywise"
   Softsign, "Apply the Softsign operator entrywise"
   Sqrt, "Apply the Sqrt operator entrywise"
   Square, "Apply the Square operator entrywise"
   SquareDifference, "Apply the SquareDifference operator entrywise"
   Subtract, "Apply the Subtract operator entrywise"
   SubtractConstant, "Apply the SubtractConstant operator entrywise"
   Tan, "Apply the Tan operator entrywise"
   Tanh, "Apply the Tanh operator entrywise"

________________________________________


.. _OperatorLayer:

----------------------------------------
OperatorLayer
----------------------------------------

OperatorLayer is composed of one or more operator objects. Operators
are applied sequentially.

Arguments:

   :ops: (``repeated Operator``)

:ref:`Back to Top<operator-layers>`

________________________________________


.. _Operator:

----------------------------------------
Operator
----------------------------------------

Operator is the base class for LBANN operators

Arguments:

   :input_type: (``lbann.DataType``) The type expected as input

   :output_type: (``lbann.DataType``) The type expected as output

   :device: (``lbann.device_allocation``) The device allocation

Methods:

   :export_proto(): Get a protobuf representation of this object

   :do_export_proto():

      Get a protobuf representation of this object

      Must be implemented in derived classes

:ref:`Back to Top<operator-layers>`

________________________________________

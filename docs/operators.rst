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

Operators only store immutable state; they do not have learnable
parameters. An operator should also be able to take objective function
gradients w.r.t. the outputs ("previous error signals") and compute
the objective function gradients w.r.t. the inputs ("error
signals"). This allows the model to perform automatic differentiation.

Operators are specified for unique input and output data types.

.. csv-table::
   :header: "Operator", "Description"
   :widths: auto

   :ref:`Operator`, "Base class for LBANN operators"
   :ref:`AbsOperator <Abs>`, "Apply the Abs operator entrywise"
   :ref:`AcoshOperator <Acosh>`, "Apply the Acosh operator entrywise"
   :ref:`AcosOperator <Acos>`, "Apply the Acos operator entrywise"
   :ref:`AddOperator <Add>`, "Apply the Add operator entrywise"
   :ref:`AddConstantOperator <AddConstant>`, "Add a constant to each input value (x+c)"
   :ref:`AsinOperator <Asin>`, "Apply the Asin operator entrywise"
   :ref:`AsinhOperator <Asinh>`, "Apply the Asinh operator entrywise"
   :ref:`AtanOperator <Atan>`, "Apply the Atan operator entrywise"
   :ref:`AtanhOperator <Atanh>`, "Apply the Atanh operator entrywise"
   :ref:`BinaryCrossEntropyOperator <BinaryCrossEntropy>`, "Apply the BinaryCrossEntropy operator entrywise"
   :ref:`BooleanAccuracyOperator <BooleanAccuracy>`, "Apply the BooleanAccuracy operator entrywise"
   :ref:`BooleanFalseNegativeOperator <BooleanFalseNegative>`, "Apply the BooleanFalseNegative operator entrywise"
   :ref:`BooleanFalsePositiveOperator <BooleanFalsePositive>`, "Apply the BooleanFalsePositive operator entrywise"
   :ref:`CeilOperator <Ceil>`, "Apply the Ceil operator entrywise"
   :ref:`ClampOperator <Clamp>`, "Constrain all values in a tensor within a range"
   :ref:`ConstantSubtractOperator <ConstantSubtract>`, "Subtract each input value from a constant (c-x)"
   :ref:`CosOperator <Cos>`, "Apply the Cos operator entrywise"
   :ref:`CoshOperator <Cosh>`, "Apply the Cosh operator entrywise"
   :ref:`DivideOperator <Divide>`, "Apply the Divide operator entrywise"
   :ref:`EqualOperator <Equal>`, "Apply the Equal operator entrywise"
   :ref:`EqualConstantOperator <EqualConstant>`, "Test each value for equality with a constant (x==c)"
   :ref:`ErfOperator <Erf>`, "Apply the Erf operator entrywise"
   :ref:`ErfInvOperator <ErfInv>`, "Apply the ErfInv operator entrywise"
   :ref:`ExpOperator <Exp>`, "Apply the Exp operator entrywise"
   :ref:`Expm1Operator <Expm1>`, "Apply the Expm1 operator entrywise"
   :ref:`FloorOperator <Floor>`, "Apply the Floor operator entrywise"
   :ref:`GeluOperator <Gelu>`, "Gaussian Error Linear Unit operator"
   :ref:`GreaterOperator <Greater>`, "Apply the Greater operator entrywise"
   :ref:`GreaterConstantOperator <GreaterConstant>`, "Test each value for 'greater-than' with a constant (x>c)"
   :ref:`GreaterEqualOperator <GreaterEqual>`, "Apply the GreaterEqual operator entrywise"
   :ref:`GreaterEqualConstantOperator <GreaterEqualConstant>`, "Test each value for 'greater-than-or-equal-to' with a constant (x>=c)"
   :ref:`LessOperator <Less>`, "Apply the Less operator entrywise"
   :ref:`LessConstantOperator <LessConstant>`, "Test each value for 'less-than' with a constant (x<c)"
   :ref:`LessEqualOperator <LessEqual>`, "Apply the LessEqual operator entrywise"
   :ref:`LessEqualConstantOperator <LessEqualConstant>`, "Test each value for 'less-than-or-equal-to' with a constant (x<=c)"
   :ref:`LogOperator <Log>`, "Apply the Log operator entrywise"
   :ref:`Log1pOperator <Log1p>`, "Apply the Log1p operator entrywise"
   :ref:`LogSigmoidOperator <LogSigmoid>`, "Apply the LogSigmoid operator entrywise"
   :ref:`LogSoftmaxOperator <LogSoftmaxOp>`, "Apply the LogSoftmax operator entrywise"
   :ref:`LogicalAndOperator <LogicalAnd>`, "Apply the LogicalAnd operator entrywise"
   :ref:`LogicalNotOperator <LogicalNot>`, "Apply the LogicalNot operator entrywise"
   :ref:`LogicalOrOperator <LogicalOr>`, "Apply the LogicalOr operator entrywise"
   :ref:`LogicalXorOperator <LogicalXor>`, "Apply the LogicalXor operator entrywise"
   :ref:`MaxOperator <Max>`, "Apply the Max operator entrywise"
   :ref:`MaxConstantOperator <MaxConstant>`, "Apply the MaxConstant operator entrywise"
   :ref:`MinOperator <Min>`, "Apply the Min operator entrywise"
   :ref:`MinConstantOperator <MinConstant>`, "Apply the MinConstant operator entrywise"
   :ref:`ModOperator <Mod>`, "Apply the Mod operator entrywise"
   :ref:`MultiplyOperator <Multiply>`, "Apply the Multiply operator entrywise"
   :ref:`NegativeOperator <Negative>`, "Apply the Log Negative entrywise"
   :ref:`NotEqualOperator <NotEqual>`, "Apply the NotEqual operator entrywise"
   :ref:`NotEqualConstantOperator <NotEqualConstant>`, "Test each value for inequality with a constant (x!=c)"
   :ref:`PowOperator <Pow>`, "Apply the Pow operator entrywise"
   :ref:`ReciprocalOperator <Reciprocal>`, "Apply the Reciprocal operator entrywise"
   :ref:`RoundOperator <Round>`, "Apply the Round operator entrywise"
   :ref:`RsqrtOperator <Rsqrt>`, "Apply the Rsqrt operator entrywise"
   :ref:`SafeDivideOperator <SafeDivide>`, "Apply the SafeDivide operator entrywise"
   :ref:`SafeReciprocalOperator <SafeReciprocal>`, "Apply the SafeReciprocal operator entrywise"
   :ref:`ScaleOperator <Scale>`, "Scale each input value by a constant value (c*x)"
   :ref:`SelectOperator <Select>`, "Chooses one input or the other based on the value of a predicate (if a return b, else c)" 
   :ref:`SeluOperator <Selu>`, "Apply the Selu operator entrywise"
   :ref:`SigmoidOperator <Sigmoid>`, "Apply the Sigmoid operator entrywise"
   :ref:`SigmoidBinaryCrossEntropyOperator <SigmoidBinaryCrossEntropy>`, "Apply the SigmoidBinaryCrossEntropy operator entrywise."
   :ref:`SignOperator <Sign>`, "Apply the Sign operator entrywise"
   :ref:`SinOperator <Sin>`, "Apply the Sin operator entrywise"
   :ref:`SinhOperator <Sinh>`, "Apply the Sinh operator entrywise"
   :ref:`SoftplusOperator <Softplus>`, "Apply the Softplus operator entrywise"
   :ref:`SoftsignOperator <Softsign>`, "Apply the Softsign operator entrywise"
   :ref:`SqrtOperator <Sqrt>`, "Apply the Sqrt operator entrywise"
   :ref:`SquareOperator <Square>`, "Apply the Square operator entrywise"
   :ref:`SquareDifferenceOperator <SquareDifference>`, "Apply the SquareDifference operator entrywise"
   :ref:`SubtractOperator <Subtract>`, "Apply the Subtract operator entrywise"
   :ref:`SubtractConstantOperator <SubtractConstant>`, "Apply the SubtractConstant operator entrywise"
   :ref:`TanOperator <Tan>`, "Apply the Tan operator entrywise"
   :ref:`TanhOperator <Tanh>`, "Apply the Tanh operator entrywise"



.. _Operator:

------------------------------------------------
Operator
------------------------------------------------

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

:ref:`Back to Top<operators>`

________________________________________________



.. _Abs:

------------------------------------------------
Abs
------------------------------------------------

Perform entrywise absolute value on the input tensor.

.. math::

  \text{Abs}(x) = |x|

:ref:`Back to Top<operators>`

________________________________________________



.. _Acosh:

------------------------------------------------
Acosh
------------------------------------------------

Apply the inverse hyperbolic cosine entrywise.

:ref:`Back to Top<operators>`

________________________________________________



.. _Acos:

------------------------------------------------
Acos
------------------------------------------------

Apply the inverse cosine function entrywise.

:ref:`Back to Top<operators>`

________________________________________________



.. _Add:

------------------------------------------------
Add
------------------------------------------------

Perform entrywise addition on two input tensors.

:ref:`Back to Top<operators>`

________________________________________________



.. _AddConstant:

------------------------------------------------
AddConstant
------------------------------------------------

Add a constant to each input value.

.. math::

   \text{AddConstant}(x,c) = x + c

Arguments:

   :constant: (``double``) The constant to be added

:ref:`Back to Top<operators>`

________________________________________________



.. _Asin:

------------------------------------------------
Asin
------------------------------------------------

Apply the inverse sine function entrywise.

:ref:`Back to Top<operators>`

________________________________________________



.. _Asinh:

------------------------------------------------
Asinh
------------------------------------------------

Apply the hyperbolic inverse sine function entrywise.

:ref:`Back to Top<operators>`

________________________________________________



.. _Atan:

------------------------------------------------
Atan
------------------------------------------------

Apply the inverse tangent function entrywise.

:ref:`Back to Top<operators>`

________________________________________________



.. _Atanh:

------------------------------------------------
Atanh
------------------------------------------------

Apply the hyperbolic inverse tangent function entrywise.

:ref:`Back to Top<operators>`

________________________________________________



.. _BinaryCrossEntropy:

------------------------------------------------
BinaryCrossEntropy
------------------------------------------------

Apply the BinaryCrossEntropy operator entrywise.

Compare each predicted probability to actual class value, either 0
or 1. Calculate the score that penalizes the probabilities based on
the distance from the expected value.

:ref:`Back to Top<operators>`

________________________________________________



.. _BooleanAccuracy:

------------------------------------------------
BooleanAccuracy
------------------------------------------------

Apply the BooleanAccuracy operator entrywise.

Applies the function:

.. math::

   \text{BooleanAccuracy}(x1,x2) = (x1 >= 0.5) == (x2 >= 0.5)

:ref:`Back to Top<operators>`

________________________________________________



.. _BooleanFalseNegative:

------------------------------------------------
BooleanFalseNegative
------------------------------------------------

Apply the BooleanFalseNegative operator entrywise.

:ref:`Back to Top<operators>`

________________________________________________



.. _BooleanFalsePositive:

------------------------------------------------
BooleanFalsePositive
------------------------------------------------

Apply the BooleanFalsePositive operator entrywise.

:ref:`Back to Top<operators>`

________________________________________________



.. _Ceil:

------------------------------------------------
Ceil
------------------------------------------------

Apply the ceiling function to an input tensor entrywise.

:ref:`Back to Top<operators>`

________________________________________________



.. _clamp:

------------------------------------------------
Clamp
------------------------------------------------

Constrain all values in a tensor within a range

.. math::

   \text{Clamp}(x; \text{min}, \text{max}) =
       \begin{cases}
         \text{min} & x \leq \text{min}           \\
         x          & \text{min} < x < \text{max} \\
         \text{max} & x \geq \text{max}
       \end{cases}

Arguments:

   :min: (``double``) Minimum value in range
   :max: (``double``) Maximum value in range

:ref:`Back to Top<operators>`

________________________________________________



.. _ConstantSubtract:

------------------------------------------------
ConstantSubtract
------------------------------------------------

Subtract each input value from a constant.

.. math::

   \text{ConstantSubtract}(c,x) = c - x

Arguments:

   :constant: (``double``) The constant to subtract from

:ref:`Back to Top<operators>`

________________________________________________



.. _Cos:

------------------------------------------------
Cos
------------------------------------------------

Compute the cosine of the input tensor entrywise.

:ref:`Back to Top<operators>`

________________________________________________



.. _Cosh:

------------------------------------------------
Cosh
------------------------------------------------

Compute the hyperbolic cosine of the input tensor entrywise.

:ref:`Back to Top<operators>`

________________________________________________



.. _Divide:

------------------------------------------------
Divide
------------------------------------------------

Perform entrywise division on two input tensors.

.. math::

   \text{Divide}(x,y) = \frac{x}{y}

:ref:`Back to Top<operators>`

________________________________________________



.. _Equal:

------------------------------------------------
Equal
------------------------------------------------

Perform entrywise logical equal on two input tensors.

:ref:`Back to Top<operators>`

________________________________________________



.. _EqualConstant:

------------------------------------------------
EqualConstant
------------------------------------------------

Perform entrywise logical equal on input tensor and a constant.

.. math::

   \text{EqualConstant}(x,c) = x \equiv c

Arguments:

   :constant: (``double``) The constant used for comparison

:ref:`Back to Top<operators>`

________________________________________________



.. _Erf:

------------------------------------------------
Erf
------------------------------------------------

Compute the error function of the inpute tensor entrywise.

:ref:`Back to Top<operators>`

________________________________________________



.. _ErfInv:

------------------------------------------------
ErfInv
------------------------------------------------

Compute the inverse error function entrywise.

:ref:`Back to Top<operators>`

________________________________________________



.. _Exp:

------------------------------------------------
Exp
------------------------------------------------

Calculate the exponential of the input tensor entrywise.

.. math::

   \text{Exp}(x) = e^x

:ref:`Back to Top<operators>`

________________________________________________



.. _Expm1:

------------------------------------------------
Expm1
------------------------------------------------

Calculate the exponential minus one of the input tensor entrywise.

.. math::

   \text{Expm1}(x) = e^x - 1

:ref:`Back to Top<operators>`

________________________________________________



.. _Floor:

------------------------------------------------
Floor
------------------------------------------------

Apply the floor function to the input tensor entrywise.

:ref:`Back to Top<operators>`

________________________________________________



.. _Gelu:

------------------------------------------------
Gelu (GELU tanh approximation)
------------------------------------------------

The Gaussian Error Linear Unit (GELU) operator is defined by:

.. math::

   \text{GELU}(x) = x\Phi (x),

where :math:`\Phi` is the Gaussian cumulative distribution function. The
hyperbolic tangent-based approximation of the Gaussian Error Linear Unit (GELU)
operator, found in the BERT and GPT transformer codebases, is implemented in
LBANN and given by:

.. math::

   \text{GELU'}(x) = \frac{x}{2} \cdot (1 + \text{tanh}(\sqrt{2 / \pi} \cdot (x + 0.044715 x^3))).

For explanation on GELU, see:

Dan Hendrycks and Kevin Gimpel. "Gaussian Error Linear Units (GELUs)."
arXiv preprint arXiv:1606.08415 (2016).

:ref:`Back to Top<operators>`

________________________________________



.. _Greater:

------------------------------------------------
Greater
------------------------------------------------

Perform entrywise logical 'greater' on two input tensors.

.. math::

   \text{Greater}(x,y) = x > y

:ref:`Back to Top<operators>`

________________________________________________



.. _GreaterConstant:

------------------------------------------------
GreaterConstant
------------------------------------------------

Perform entrywise logical 'greater-than' on input tensor and a constant.

.. math::

   \text{GreaterConstant}(x,c) = x > c

Arguments:

   :constant: (``double``) The constant to be used for comparison

:ref:`Back to Top<operators>`

________________________________________________



.. _GreaterEqual:

------------------------------------------------
GreaterEqual
------------------------------------------------

Perform entrywise logical 'greater-or-equal' on two input tensors.

.. math::

   \text{GreaterEqual}(x,y) = x \geq y

:ref:`Back to Top<operators>`

________________________________________________



.. _GreaterEqualConstant:

------------------------------------------------
GreaterEqualConstant
------------------------------------------------

Perform entrywise logical 'greater-or-equal' on input tensor and a
constant.

.. math::

   \text{GreaterEqualConstant}(x,c) = x \geq c

Arguments:

   :constant: (``double``) The constant to be used for comparison

:ref:`Back to Top<operators>`

________________________________________________



.. _Less:

------------------------------------------------
Less
------------------------------------------------

Perform entrywise logical 'less-than' on two input tensors.

.. math::

   \text{Less}(x,y) = x < y

:ref:`Back to Top<operators>`

________________________________________________



.. _LessConstant:

------------------------------------------------
LessConstant
------------------------------------------------

Perform entrywise logical 'less-than' on input tensor and a constant.

.. math::

   \text{LessConstant}(x,y) = x < c

Arguments:

   :constant: (``double``) The constant to be used for comparison

:ref:`Back to Top<operators>`

________________________________________________



.. _LessEqual:

------------------------------------------------
LessEqual
------------------------------------------------

Perform entrywise logical 'less-equal' on two input tensors.

.. math::

   \text{LessEqual}(x,y) = x \leq y

:ref:`Back to Top<operators>`

________________________________________________



.. _LessEqualConstant:

------------------------------------------------
LessEqualConstant
------------------------------------------------

Perform entrywise logical 'less-or-equal' on input tensor and a
constant.

.. math::

   \text{LessEqualConstant}(x,c) = x \leq c

Arguments:

   :constant: (``double``) The constant to be used for comparison

:ref:`Back to Top<operators>`

________________________________________________



.. _Log:

------------------------------------------------
Log
------------------------------------------------

Calculate the log of the input tensor entrywise.

:ref:`Back to Top<operators>`

________________________________________________



.. _Log1p:

------------------------------------------------
Log1p
------------------------------------------------

Calculate the log of one plus the input tensor entrywise.

.. math::

   \text{Log1p}(x) = \log{1 + x}

:ref:`Back to Top<operators>`

________________________________________________



.. _LogSigmoid:

------------------------------------------------
LogSigmoid
------------------------------------------------

Calculate the log of the output from the sigmoid function entrywise.

.. math::

   \text{LogSigmoid}(x) = \log \frac{1}{1+e^{-x}}

:ref:`Back to Top<operators>`

________________________________________________



.. _LogSoftmaxOp:

------------------------------------------------
LogSoftmax
------------------------------------------------

Calculate the log of the softmax function entrywise.

.. math::

   \text{LogSoftmax}(x)_i = x_i - \log \sum_j e^{x_j}

:ref:`Back to Top<operators>`

________________________________________________

.. _LogicalAnd:

------------------------------------------------
LogicalAnd
------------------------------------------------

Perform entrywise logical 'and' on two input tensors.

:ref:`Back to Top<operators>`

________________________________________________



.. _LogicalNot:

------------------------------------------------
LogicalNot
------------------------------------------------

Perform entrywise logical 'not' on two input tensors.

:ref:`Back to Top<operators>`

________________________________________________



.. _LogicalOr:

------------------------------------------------
LogicalOr
------------------------------------------------

Perform entrywise logical 'or' on two input tensors.

:ref:`Back to Top<operators>`

________________________________________________



.. _LogicalXor:

------------------------------------------------
LogicalXor
------------------------------------------------

Perform entrywise logical 'xor' on two input tensors.

:ref:`Back to Top<operators>`

________________________________________________



.. _Max:

------------------------------------------------
Max
------------------------------------------------

Perform entrywise max of input tensors.

:ref:`Back to Top<operators>`

________________________________________________



.. _MaxConstant:

------------------------------------------------
MaxConstant
------------------------------------------------

Perform entrywise max of input tensor against a constant.

:ref:`Back to Top<operators>`

________________________________________________



.. _Min:

------------------------------------------------
Min
------------------------------------------------

Perform entrywise min of input tensors.

:ref:`Back to Top<operators>`

________________________________________________



.. _MinConstant:

------------------------------------------------
MinConstant
------------------------------------------------

Perform entrywise min of input tensor against a constant.

:ref:`Back to Top<operators>`

________________________________________________



.. _Mod:

------------------------------------------------
Mod
------------------------------------------------

Perform entrywise modulus on two input tensors.

:ref:`Back to Top<operators>`

________________________________________________



.. _Multiply:

------------------------------------------------
Multiply
------------------------------------------------

Perform entrywise multiplication on input tensors.

:ref:`Back to Top<operators>`

________________________________________________



.. _Negative:

------------------------------------------------
Negative
------------------------------------------------

Produce output tensor with flipped sign.

.. math::

   \text{Negative}(x) = -x

:ref:`Back to Top<operators>`

________________________________________________



.. _NotEqual:

------------------------------------------------
NotEqual
------------------------------------------------

Perform entrywise logical 'not-equal' on two input tensors.

:ref:`Back to Top<operators>`

________________________________________________



.. _NotEqualConstant:

------------------------------------------------
NotEqualConstant
------------------------------------------------

Perform entrywise logical 'not-equal' on input tensor and a constant.

.. math::

   \text{NotEqualConstant}(x, c) = x \neq c

Arguments:

   :constant: (``double``) The constant to be used for comparison

:ref:`Back to Top<operators>`

________________________________________________



.. _Pow:

------------------------------------------------
Pow
------------------------------------------------

Perform entrywise exponent using one input tensor as the base and a
second input tensor as the exponent.

.. math::

   \text{Pow}(x,y) = x^y

:ref:`Back to Top<operators>`

________________________________________________



.. _Reciprocal:

------------------------------------------------
Reciprocal
------------------------------------------------

Perform entrywise reciprocal function on input tensor.

.. math::

   \text{Reciprocal}(x) = \frac{1}{x}

:ref:`Back to Top<operators>`

________________________________________________



.. _Round:

------------------------------------------------
Round
------------------------------------------------

Round input tensor values to the nearest integer entrywise.

:ref:`Back to Top<operators>`

________________________________________________



.. _Rsqrt:

------------------------------------------------
Rsqrt
------------------------------------------------

Compute reciprocal of square-root of values in the input tensor
entrywise.

.. math::

   \text{Rsqrt}(x) = \frac{1}{\sqrt{x}}

:ref:`Back to Top<operators>`

________________________________________________



.. _SafeDivide:

------------------------------------------------
SafeDivide
------------------------------------------------

FIXME: Is this right?

Perform entrywise division on two input tensors. Return zero if the
divisor is zero.

:ref:`Back to Top<operators>`

________________________________________________



.. _SafeReciprocal:

------------------------------------------------
SafeReciprocal
------------------------------------------------

FIXME: Is this right?

Perform entrywise reciprocal function on input tensor. Return zero if
the input value is zero.

.. math::

   \text{SafeReciprocal}(x) = \frac{1}{x}

:ref:`Back to Top<operators>`

________________________________________________



.. _Scale:

------------------------------------------------
Scale
------------------------------------------------

Scale each input value by a constant.

.. math::

   \text{Scale}(x,c) = c * x

Arguments:

   :constant: (``double``) The constant to scale by

:ref:`Back to Top<operators>`

________________________________________________



.. _Select:

------------------------------------------------
Select
------------------------------------------------

Chooses one input or the other based on the value of a predicate (if a return b,
else c). The predicate is given as the first tensor, followed by the ``true``
and ``false`` results, respectively. To optimize the operator for comparison
predicates, the ``value`` and ``epsilon`` arguments provide a value to compare
with.

For further optimization, the true and false tensors can be replaced
with a constant value via the arguments (see below). If one of those values (or
both) are set, the respective tensor parameters are not necessary. For example,
``Select(condition, some_tensor, value=1, if_true=2)`` in the Python frontend
will internally set ``constant_if_true`` to ``True`` and ``value_if_true`` to
``2``. The resulting expression would be ``(condition == 1) ? 2 : some_tensor``
for every input element.

In general, the following statement is executed:


.. math::

   \text{Select}(pred, trueval, falseval) =
       \begin{cases}
         trueval  & | pred - \text{value} | < \text{epsilon} \\
         falseval & otherwise
       \end{cases}



Arguments:

   :value: (``double``) The value to compare the predicate with
   :epsilon: (``double``) Comparison threshold (default: 1e-5)
   :constant_if_true: (``bool``) If true, uses ``value_if_true`` as a constant true value
   :constant_if_false: (``bool``) If true, uses ``value_if_false`` as a constant false value
   :value_if_true: (``double``) If set, uses the given value instead of the second parameter
   :value_if_false: (``double``) If set, uses the given value instead of the third parameter

:ref:`Back to Top<operators>`

________________________________________________



.. _Selu:

------------------------------------------------
Selu
------------------------------------------------

Apply scaled exponential linear unit function to input tensor
entrywise.

:ref:`Back to Top<operators>`

________________________________________________



.. _Sigmoid:

------------------------------------------------
Sigmoid
------------------------------------------------

Apply the sigmoid function to the input tensor entrywise.

.. math::

   \text{Sigmoid}(x) = \frac{1}{1+e^{-x}}

:ref:`Back to Top<operators>`

________________________________________________



.. _SigmoidBinaryCrossEntropy:

------------------------------------------------
SigmoidBinaryCrossEntropy
------------------------------------------------

FIXME: Better description of this?

Apply the SigmoidBinaryCrossEntropy operator entrywise.

:ref:`Back to Top<operators>`

________________________________________________



.. _Sign:

------------------------------------------------
Sign
------------------------------------------------

FIXME: Is this output right?

Compute the sign of the imput tensor entrywise. If input > 0,
output 1. if input < 0, output -1. if input == 0, output 0.

:ref:`Back to Top<operators>`

________________________________________________



.. _Sin:

------------------------------------------------
Sin
------------------------------------------------

Calculate entrywise sine of the input tensor.


:ref:`Back to Top<operators>`

________________________________________________



.. _Sinh:

------------------------------------------------
Sinh
------------------------------------------------

Calculate entrywise hyperbolic sine of the input tensor.

:ref:`Back to Top<operators>`

________________________________________________



.. _Softplus:

------------------------------------------------
Softplus
------------------------------------------------

Calculate the softplus of the input tensor entrywise.

.. math::

   \text{Softplus}(x) = \log{1 + e^x}

:ref:`Back to Top<operators>`

________________________________________________



.. _Softsign:

------------------------------------------------
Softsign
------------------------------------------------

Calculate the softsign of the input tensor entrywise.

.. math::

   \text{Softsign}(x) = \frac{x}{1+|x|}

:ref:`Back to Top<operators>`

________________________________________________



.. _Sqrt:

------------------------------------------------
Sqrt
------------------------------------------------

Compute square root of input tensor values entrywise.

:ref:`Back to Top<operators>`

________________________________________________



.. _Square:

------------------------------------------------
Square
------------------------------------------------

Compute square of input tensor values entrywise.

:ref:`Back to Top<operators>`

________________________________________________



.. _SquareDifference:

------------------------------------------------
SquareDifference
------------------------------------------------

FIXME: Better description?

Apply the SquareDifference operator entrywise

:ref:`Back to Top<operators>`

________________________________________________



.. _subtract:

------------------------------------------------
Subtract
------------------------------------------------

Perform entrywise subtraction on two input tensors.

.. math::

   \text{Subtract}(x,y) = x - y


:ref:`Back to Top<operators>`

________________________________________________



.. _SubtractConstant:

------------------------------------------------
SubtractConstant
------------------------------------------------

Subtract a constant from from the input tensor entrywise.

.. math::

   \text{SubtractConstant}(x,c) = x - c

Arguments:

   :constant: (``double``) The constant to subtract

:ref:`Back to Top<operators>`

________________________________________________



.. _Tan:

------------------------------------------------
Tan
------------------------------------------------

Apply the tangent function entrywise.

:ref:`Back to Top<operators>`

________________________________________________



.. _Tanh:

------------------------------------------------
Tanh
------------------------------------------------

Apply the hyperbolic tangent function entrywise.

:ref:`Back to Top<operators>`

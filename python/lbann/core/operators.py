"""LBANN Operators.

Operators are atomic tensor operations supported by LBANN.

"""

from __future__ import annotations
from typing import Optional

from lbann import operators_pb2 as OpProto
from lbann import DataType, DeviceAllocation
from lbann.util import make_iterable

class Operator:
    """Base class for LBANN operators"""
    def __init__(self,
                 input_type: DataType = DataType.FLOAT,
                 output_type: DataType = None,
                 device: DeviceAllocation = None):
        """Construct an operator.

        Args:
            input_type: The type expected as input.
            output_type: The type expected as output.
            device: The device allocation.
        """
        if output_type is None:
            output_type = input_type
        self.input_type = input_type
        self.output_type = output_type
        self.device = device

    def export_proto(self):
        """Get a protobuf representation of this object."""

        op = OpProto.Operator()
        op.input_datatype = self.input_type
        op.output_datatype = self.output_type
        if self.device:
            op.device_allocation = self.device
        op.parameters.Pack(self.do_export_proto())
        return op

    def do_export_proto(self):
        """Get a protobuf representation of this object.

        Must be implemented in derived classes.
        """
        raise NotImplementedError

class Clamp(Operator):
    """Constrain all values in a tensor within a range."""
    def __init__(self,
                 input_type: DataType = DataType.FLOAT,
                 output_type: DataType = None,
                 device: str = None,
                 min: float = 0.0,
                 max: float = 1.0):
        super().__init__(input_type, output_type, device)
        self.min = min
        self.max = max

    def do_export_proto(self):
        params = OpProto.ClampOperator()
        params.min = self.min
        params.max = self.max
        return params

class Abs(Operator):
    """Apply the Abs operator entrywise."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def do_export_proto(self):
        params = OpProto.AbsOperator()
        return params

class Acos(Operator):
    """Apply the Acos operator entrywise."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def do_export_proto(self):
        params = OpProto.AcosOperator()
        return params

class Acosh(Operator):
    """Apply the Acosh operator entrywise."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def do_export_proto(self):
        params = OpProto.AcoshOperator()
        return params

class Add(Operator):
    """Apply the Add operator entrywise."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def do_export_proto(self):
        params = OpProto.AddOperator()
        return params

class AddConstant(Operator):
    """Add a constant to each input value (x+c)."""
    def __init__(self, constant: float = 0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.constant = constant

    def do_export_proto(self):
        params = OpProto.AddConstantOperator()
        params.constant = self.constant
        return params

class Asin(Operator):
    """Apply the Asin operator entrywise."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def do_export_proto(self):
        params = OpProto.AsinOperator()
        return params

class Asinh(Operator):
    """Apply the Asinh operator entrywise."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def do_export_proto(self):
        params = OpProto.AsinhOperator()
        return params

class Atan(Operator):
    """Apply the Atan operator entrywise."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def do_export_proto(self):
        params = OpProto.AtanOperator()
        return params

class Atanh(Operator):
    """Apply the Atanh operator entrywise."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def do_export_proto(self):
        params = OpProto.AtanhOperator()
        return params

class BinaryCrossEntropy(Operator):
    """Apply the BinaryCrossEntropy operator entrywise."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def do_export_proto(self):
        params = OpProto.BinaryCrossEntropyOperator()
        return params

class BooleanAccuracy(Operator):
    """Apply the BooleanAccuracy operator entrywise."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def do_export_proto(self):
        params = OpProto.BooleanAccuracyOperator()
        return params

class BooleanFalseNegative(Operator):
    """Apply the BooleanFalseNegative operator entrywise."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def do_export_proto(self):
        params = OpProto.BooleanFalseNegativeOperator()
        return params

class BooleanFalsePositive(Operator):
    """Apply the BooleanFalsePositive operator entrywise."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def do_export_proto(self):
        params = OpProto.BooleanFalsePositiveOperator()
        return params

class Ceil(Operator):
    """Apply the Ceil operator entrywise."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def do_export_proto(self):
        params = OpProto.CeilOperator()
        return params

class ConstantSubtract(Operator):
    """Subtract each input value from a constant (c-x)."""
    def __init__(self, constant: float = 0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.constant = constant

    def do_export_proto(self):
        params = OpProto.ConstantSubtractOperator()
        params.constant = self.constant
        return params

class Cos(Operator):
    """Apply the Cos operator entrywise."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def do_export_proto(self):
        params = OpProto.CosOperator()
        return params

class Cosh(Operator):
    """Apply the Cosh operator entrywise."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def do_export_proto(self):
        params = OpProto.CoshOperator()
        return params

class Divide(Operator):
    """Apply the Divide operator entrywise."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def do_export_proto(self):
        params = OpProto.DivideOperator()
        return params

class Equal(Operator):
    """Apply the Equal operator entrywise."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def do_export_proto(self):
        params = OpProto.EqualOperator()
        return params

class EqualConstant(Operator):
    """Test each value for equality with a constant (x==c)."""
    def __init__(self, constant: float = 0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.constant = constant

    def do_export_proto(self):
        params = OpProto.EqualConstantOperator()
        params.constant = self.constant
        return params

class Erf(Operator):
    """Apply the Erf operator entrywise."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def do_export_proto(self):
        params = OpProto.ErfOperator()
        return params

class ErfInv(Operator):
    """Apply the ErfInv operator entrywise."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def do_export_proto(self):
        params = OpProto.ErfInvOperator()
        return params

class Exp(Operator):
    """Apply the Exp operator entrywise."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def do_export_proto(self):
        params = OpProto.ExpOperator()
        return params

class Expm1(Operator):
    """Apply the Expm1 operator entrywise."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def do_export_proto(self):
        params = OpProto.Expm1Operator()
        return params

class Floor(Operator):
    """Apply the Floor operator entrywise."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def do_export_proto(self):
        params = OpProto.FloorOperator()
        return params

class Gelu(Operator):
    """Apply the GELU operator entrywise."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def do_export_proto(self):
        params = OpProto.GeluOperator()
        return params

class Greater(Operator):
    """Apply the Greater operator entrywise."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def do_export_proto(self):
        params = OpProto.GreaterOperator()
        return params

class GreaterConstant(Operator):
    """Test each value for "greater-than" with a constant (x>c)."""
    def __init__(self, constant: float = 0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.constant = constant

    def do_export_proto(self):
        params = OpProto.GreaterConstantOperator()
        params.constant = self.constant
        return params

class GreaterEqual(Operator):
    """Apply the GreaterEqual operator entrywise."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def do_export_proto(self):
        params = OpProto.GreaterEqualOperator()
        return params

class GreaterEqualConstant(Operator):
    """Test each value for "greater-than-or-equal-to" with a constant (x>=c)."""
    def __init__(self, constant: float = 0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.constant = constant

    def do_export_proto(self):
        params = OpProto.GreaterEqualConstantOperator()
        params.constant = self.constant
        return params

class Less(Operator):
    """Apply the Less operator entrywise."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def do_export_proto(self):
        params = OpProto.LessOperator()
        return params

class LessConstant(Operator):
    """Test each value for "less-than" with a constant (x<c)."""
    def __init__(self, constant: float = 0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.constant = constant

    def do_export_proto(self):
        params = OpProto.LessConstantOperator()
        params.constant = self.constant
        return params

class LessEqual(Operator):
    """Apply the LessEqual operator entrywise."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def do_export_proto(self):
        params = OpProto.LessEqualOperator()
        return params

class LessEqualConstant(Operator):
    """Test each value for "less-than-or-equal-to with a constant (x<=c)."""
    def __init__(self, constant: float = 0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.constant = constant

    def do_export_proto(self):
        params = OpProto.LessEqualConstantOperator()
        params.constant = self.constant
        return params

class Log(Operator):
    """Apply the Log operator entrywise."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def do_export_proto(self):
        params = OpProto.LogOperator()
        return params

class Log1p(Operator):
    """Apply the Log1p operator entrywise."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def do_export_proto(self):
        params = OpProto.Log1pOperator()
        return params

class LogSigmoid(Operator):
    """Apply the LogSigmoid operator entrywise."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def do_export_proto(self):
        params = OpProto.LogSigmoidOperator()
        return params

class LogicalAnd(Operator):
    """Apply the LogicalAnd operator entrywise."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def do_export_proto(self):
        params = OpProto.LogicalAndOperator()
        return params

class LogicalNot(Operator):
    """Apply the LogicalNot operator entrywise."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def do_export_proto(self):
        params = OpProto.LogicalNotOperator()
        return params

class LogicalOr(Operator):
    """Apply the LogicalOr operator entrywise."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def do_export_proto(self):
        params = OpProto.LogicalOrOperator()
        return params

class LogicalXor(Operator):
    """Apply the LogicalXor operator entrywise."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def do_export_proto(self):
        params = OpProto.LogicalXorOperator()
        return params

class Max(Operator):
    """Apply the Max operator entrywise."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def do_export_proto(self):
        params = OpProto.MaxOperator()
        return params

class MaxConstant(Operator):
    """Perform entrywise max of input tensor against a constant."""
    def __init__(self, constant: float = 0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.constant = constant

    def do_export_proto(self):
        params = OpProto.MaxConstantOperator()
        params.constant = self.constant
        return params

class Min(Operator):
    """Apply the Min operator entrywise."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def do_export_proto(self):
        params = OpProto.MinOperator()
        return params

class MinConstant(Operator):
    """Perform entrywise min of input tensor against a constant."""
    def __init__(self, constant: float = 0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.constant = constant

    def do_export_proto(self):
        params = OpProto.MinConstantOperator()
        params.constant = self.constant
        return params

class Mod(Operator):
    """Apply the Mod operator entrywise."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def do_export_proto(self):
        params = OpProto.ModOperator()
        return params

class Multiply(Operator):
    """Apply the Multiply operator entrywise."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def do_export_proto(self):
        params = OpProto.MultiplyOperator()
        return params

class Negative(Operator):
    """Apply the Negative operator entrywise."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def do_export_proto(self):
        params = OpProto.NegativeOperator()
        return params

class NotEqual(Operator):
    """Apply the NotEqual operator entrywise."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def do_export_proto(self):
        params = OpProto.NotEqualOperator()
        return params

class NotEqualConstant(Operator):
    """Test each value for inequality with a constant (x!=c)."""
    def __init__(self, constant: float = 0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.constant = constant

    def do_export_proto(self):
        params = OpProto.NotEqualConstantOperator()
        params.constant = self.constant
        return params

class Pow(Operator):
    """Apply the Pow operator entrywise."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def do_export_proto(self):
        params = OpProto.PowOperator()
        return params

class Reciprocal(Operator):
    """Apply the Reciprocal operator entrywise."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def do_export_proto(self):
        params = OpProto.ReciprocalOperator()
        return params

class Round(Operator):
    """Apply the Round operator entrywise."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def do_export_proto(self):
        params = OpProto.RoundOperator()
        return params

class Rsqrt(Operator):
    """Apply the Rsqrt operator entrywise."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def do_export_proto(self):
        params = OpProto.RsqrtOperator()
        return params

class SafeDivide(Operator):
    """Apply the SafeDivide operator entrywise."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def do_export_proto(self):
        params = OpProto.SafeDivideOperator()
        return params

class SafeReciprocal(Operator):
    """Apply the SafeReciprocal operator entrywise."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def do_export_proto(self):
        params = OpProto.SafeReciprocalOperator()
        return params

class Scale(Operator):
    """Scale each input value by a constant value (c*x)."""
    def __init__(self, constant: float = 0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.constant = constant

    def do_export_proto(self):
        params = OpProto.ScaleOperator()
        params.constant = self.constant
        return params

class Select(Operator):
    """
    Select one tensor (or value) or another based on a predicate. The
    predicate is an equality predicate for optimization purposes.
    """
    def __init__(self, *args, value: float = 0.0, epsilon: float = 1e-5,
                 if_true: Optional[float] = None,
                 if_false: Optional[float] = None, **kwargs):
        super().__init__(*args, **kwargs)

        self.value = value
        self.epsilon = epsilon

        # Setup optional properties
        if if_true is not None:
            self.constant_if_true = True
            self.value_if_true = if_true
        else:
            self.constant_if_true = False
            self.value_if_true = 0.0

        if if_false is not None:
            self.constant_if_false = True
            self.value_if_false = if_false
        else:
            self.constant_if_false = False
            self.value_if_false = 0.0

    def do_export_proto(self):
        params = OpProto.SelectOperator()
        params.value = self.value
        params.epsilon = self.epsilon
        params.constant_if_true = self.constant_if_true
        params.constant_if_false = self.constant_if_false
        params.value_if_true = self.value_if_true
        params.value_if_false = self.value_if_false
        return params

class Selu(Operator):
    """Apply the Selu operator entrywise."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def do_export_proto(self):
        params = OpProto.SeluOperator()
        return params

class Sigmoid(Operator):
    """Apply the Sigmoid operator entrywise."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def do_export_proto(self):
        params = OpProto.SigmoidOperator()
        return params

class SigmoidBinaryCrossEntropy(Operator):
    """Apply the SigmoidBinaryCrossEntropy operator entrywise."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def do_export_proto(self):
        params = OpProto.SigmoidBinaryCrossEntropyOperator()
        return params

class Sign(Operator):
    """Apply the Sign operator entrywise."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def do_export_proto(self):
        params = OpProto.SignOperator()
        return params

class Sin(Operator):
    """Apply the Sin operator entrywise."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def do_export_proto(self):
        params = OpProto.SinOperator()
        return params

class Sinh(Operator):
    """Apply the Sinh operator entrywise."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def do_export_proto(self):
        params = OpProto.SinhOperator()
        return params

class Softplus(Operator):
    """Apply the Softplus operator entrywise."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def do_export_proto(self):
        params = OpProto.SoftplusOperator()
        return params

class Softsign(Operator):
    """Apply the Softsign operator entrywise."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def do_export_proto(self):
        params = OpProto.SoftsignOperator()
        return params

class Sqrt(Operator):
    """Apply the Sqrt operator entrywise."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def do_export_proto(self):
        params = OpProto.SqrtOperator()
        return params

class Square(Operator):
    """Apply the Square operator entrywise."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def do_export_proto(self):
        params = OpProto.SquareOperator()
        return params

class SquaredDifference(Operator):
    """Apply the SquaredDifference operator entrywise."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def do_export_proto(self):
        params = OpProto.SquaredDifferenceOperator()
        return params

class Subtract(Operator):
    """Apply the Subtract operator entrywise."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def do_export_proto(self):
        params = OpProto.SubtractOperator()
        return params

class SubtractConstant(Operator):
    """Subtract a constant from each input value (x-c)."""
    def __init__(self, constant: float = 0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.constant = constant

    def do_export_proto(self):
        params = OpProto.SubtractConstantOperator()
        params.constant = self.constant
        return params

class Tan(Operator):
    """Apply the Tan operator entrywise."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def do_export_proto(self):
        params = OpProto.TanOperator()
        return params

class Tanh(Operator):
    """Apply the Tanh operator entrywise."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def do_export_proto(self):
        params = OpProto.TanhOperator()
        return params

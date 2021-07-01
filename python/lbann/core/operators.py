"""LBANN Operators.

Operators are atomic tensor operations supported by LBANN.

"""

from __future__ import annotations

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
                 val_min: float = 0.0,
                 val_max: float = 1.0):
        super().__init__(input_type, output_type, device)
        self.min = val_min
        self.max = val_max

    def do_export_proto(self):
        params = OpProto.ClampOperator()
        params.min = self.min
        params.max = self.max
        return params

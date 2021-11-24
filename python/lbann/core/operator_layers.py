"""Aliases for single-operator operator layers.

This is for backward-compatibility with the current PFE. My guess/hope
is that these will be removed when the PFE attains version 1.0.
"""

from __future__ import annotations

import inspect

from lbann.core.layer import OperatorLayer
import lbann.core.operators

def generate_operator_layer(operator_class):
    """Create operator layer class for a single operator

    Returns a class that inherits from lbann.OperatorLayer.

    Args:
        operator_class (type): A derived class of
            lbann.operators.Operator

    """

    def __init__(self, *args, **kwargs):
        """Operator layer with a single operator

        Forwards arguments to lbann.OperatorLayer or sub-class of
        lbann.Operator.

        """
        layer_kwargs = lbann.Layer.__init__.__kwdefaults__.copy()
        op_kwargs = {}
        for key, value in kwargs.items():
            if key in layer_kwargs:
                layer_kwargs[key] = value
            else:
                op_kwargs[key] = value
        layer_kwargs['ops'] = [ operator_class(**op_kwargs) ]
        OperatorLayer.__init__(self, *args, **layer_kwargs)

    def export_proto(self):
        """Construct and return a protobuf message."""

        # Use default datatype if not specified
        if self.datatype is None:
            self.datatype = 0

        # Convert device string to enum
        device = lbann.DeviceAllocation.DEFAULT_DEVICE
        if isinstance(self.device, str):
            if self.device.lower() == 'cpu':
                device = lbann.DeviceAllocation.CPU
            elif self.device.lower() == 'gpu':
                device = lbann.DeviceAllocation.GPU

        # Configure operators to match layer
        for o in self.ops:
            o.input_type = self.datatype
            o.output_type = self.datatype
            o.device_allocation = device

        # Generate Protobuf message
        return OperatorLayer.export_proto(self)

    # Return operator layer class
    class_name = operator_class.__name__
    class_dict = {'__init__': __init__, 'export_proto': export_proto}
    return type(class_name, (OperatorLayer,), class_dict)

def is_operator_class(obj):
    return inspect.isclass(obj) and issubclass(obj, lbann.core.operators.Operator) and obj is not lbann.core.operators.Operator

# Generate operator layer classes based on operator classes
ops_classes = inspect.getmembers(lbann.core.operators, is_operator_class)
for op in ops_classes:
    op_name, op_class = op
    globals()[op_name] = generate_operator_layer(op_class)

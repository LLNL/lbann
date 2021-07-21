"""Aliases for single-operator operator layers.

This is for backward-compatibility with the current PFE. My guess/hope
is that these will be removed when the PFE attains version 1.0.
"""

from __future__ import annotations

import inspect

from lbann.core.layer import OperatorLayer
import lbann.core.operators

def generate_operator_layer(operator_class):

    def create_layer(*args, **kwargs):
        # Yeahhhh this seems like a GREAT idea... But it honestly
        # seems dumber to copy the list from layers.py, thereby
        # creating two glaring maintenance issues where there is
        # currently only one.
        layer_keys = lbann.Layer.__init__.__kwdefaults__.keys()
        layer_kwargs = { k: v for k,v in kwargs.items() if k in layer_keys }
        op_kwargs = { k: v for k,v in kwargs.items() if k not in layer_keys }

        layer_kwargs['ops'] = [ operator_class(**op_kwargs) ]
        return OperatorLayer(*args, **layer_kwargs)

    return create_layer

def is_operator_class(obj):
    return inspect.isclass(obj) and issubclass(obj, lbann.core.operators.Operator) and obj is not lbann.core.operators.Operator

ops_classes = inspect.getmembers(lbann.core.operators, is_operator_class)
for op in ops_classes:
    op_name, op_class = op
    globals()[op_name] = generate_operator_layer(op_class)

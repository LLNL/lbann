"""Utility functions for core LBANN functionality.

This submodule mostly contains helper functions to generate classes
from Protobuf messages.

"""
import google.protobuf.descriptor
import google.protobuf.wrappers_pb2
from lbann import lbann_pb2, callbacks_pb2, datatype_pb2, layers_pb2, metrics_pb2, model_pb2, objective_functions_pb2, optimizers_pb2, weights_pb2
from lbann.util import make_iterable

# Each field in a Protobuf message is labeled as 'optional',
# 'required', or 'repeated'
# Note: 'optional' is not used in Protobuf 3.
_protobuf_field_label_names = {
    google.protobuf.descriptor.FieldDescriptor.LABEL_OPTIONAL: 'optional',
    google.protobuf.descriptor.FieldDescriptor.LABEL_REQUIRED: 'required',
    google.protobuf.descriptor.FieldDescriptor.LABEL_REPEATED: 'repeated'
}

# Each field in a Protobuf message has a type, e.g. float, int64
_protobuf_field_type_names = {
    google.protobuf.descriptor.FieldDescriptor.TYPE_BOOL: 'bool',
    google.protobuf.descriptor.FieldDescriptor.TYPE_BYTES: 'bytes',
    google.protobuf.descriptor.FieldDescriptor.TYPE_DOUBLE: 'double',
    google.protobuf.descriptor.FieldDescriptor.TYPE_ENUM: 'enum',
    google.protobuf.descriptor.FieldDescriptor.TYPE_FIXED32: 'fixed32',
    google.protobuf.descriptor.FieldDescriptor.TYPE_FIXED64: 'fixed64',
    google.protobuf.descriptor.FieldDescriptor.TYPE_FLOAT: 'float',
    google.protobuf.descriptor.FieldDescriptor.TYPE_GROUP: 'group',
    google.protobuf.descriptor.FieldDescriptor.TYPE_INT32: 'int32',
    google.protobuf.descriptor.FieldDescriptor.TYPE_INT64: 'int64',
    google.protobuf.descriptor.FieldDescriptor.TYPE_MESSAGE: 'message',
    google.protobuf.descriptor.FieldDescriptor.TYPE_SFIXED32: 'sfixed32',
    google.protobuf.descriptor.FieldDescriptor.TYPE_SFIXED64: 'sfixed64',
    google.protobuf.descriptor.FieldDescriptor.TYPE_SINT32: 'sint32',
    google.protobuf.descriptor.FieldDescriptor.TYPE_SINT64: 'sint64',
    google.protobuf.descriptor.FieldDescriptor.TYPE_STRING: 'string',
    google.protobuf.descriptor.FieldDescriptor.TYPE_UINT32: 'uint32',
    google.protobuf.descriptor.FieldDescriptor.TYPE_UINT64: 'uint64'
}

# Wrapper Protobuf messages for primitive types
# Note: Protobuf 3 does not support optional message fields with
# primitive types. If a primitive field is not set, its value is
# "zero" (false for bool, empty string for string, etc). We need to
# use these wrapper messages to distinguish between values that are
# "zero" and values that are not set.
_protobuf_type_wrappers = (
    google.protobuf.wrappers_pb2.DoubleValue.DESCRIPTOR,
    google.protobuf.wrappers_pb2.FloatValue.DESCRIPTOR,
    google.protobuf.wrappers_pb2.Int64Value.DESCRIPTOR,
    google.protobuf.wrappers_pb2.Int64Value.DESCRIPTOR,
    google.protobuf.wrappers_pb2.UInt64Value.DESCRIPTOR,
    google.protobuf.wrappers_pb2.Int32Value.DESCRIPTOR,
    google.protobuf.wrappers_pb2.UInt32Value.DESCRIPTOR,
    google.protobuf.wrappers_pb2.BoolValue.DESCRIPTOR,
    google.protobuf.wrappers_pb2.StringValue.DESCRIPTOR,
    google.protobuf.wrappers_pb2.BytesValue.DESCRIPTOR
)

def _generate_class(message_descriptor,
                    base_field_name,
                    base_class,
                    base_kwargs,
                    base_has_export_proto):
    """Generate new class from Protobuf message.

    Args:
        message (google.protobuf.descriptor.Descriptor): Descriptor
            for Protobuf message.
        base_field_name (str): Name of corresponding field in parent
            message.
        base_class (type): Base class for generated class.
        base_kwargs (Iterable of str): Keyword arguments for base
            class `__init__` method.
        base_has_export_proto (bool): Whether the base class
            implements an `export_proto` method. If `True`, the
            generated class `export_proto` will set the appropriate
            field in the Protobuf message returned by the base class
            `export_proto`.

    Returns:
        type: Generated class.

    """

    # Names of Protobuf message and its fields
    message_name = message_descriptor.name
    field_descriptors = message_descriptor.fields_by_name
    field_names = field_descriptors.keys()
    enums = message_descriptor.enum_types_by_name
        # Handle "enum" type data.
    all_enums = {}
    for enum_name, enum_desc in enums.items():
        enum_val_to_num = {}
        enum_val_descs = enum_desc.values_by_name
        for val_name, val_desc in enum_val_descs.items():
            enum_val_to_num[val_name] = val_desc.number
        all_enums[enum_name] = type(enum_name, (), enum_val_to_num)
    # Note (trb 12/18/19): This is *NOT* meant to be a rigorous enum
    # implementation (see the 'enum' module for such a thing). The
    # goal is to simply expose "enum-like" semantics to the Python
    # front-end:
    #
    #   x = ClassName.EnumName.ENUM_VALUE
    #
    # Note that the value held by "x" after this will be "int". Based
    # on my testing, Protobuf message classes are happy enough to take
    # their enum-valued field values as "int", so this is not a
    # problem.

    # Make sure fields in generated and base classes are distinct
    for arg in base_kwargs:
        if arg in field_names:
            raise RuntimeError(
                'class {0} and its parent class {1} '
                'both define the field {2}. This is a bug!'
                .format(message_name, base_class.__name__, arg))

    def __init__(self, *args, **kwargs):

        # Extract arguments to pass to base class constructor
        _base_kwargs = {}
        for arg in base_kwargs:
            if arg in kwargs:
                _base_kwargs[arg] = kwargs[arg]
                del kwargs[arg]
        base_class.__init__(self, *args, **_base_kwargs)

        # Make sure arguments are valid
        for arg in kwargs:
            if arg not in field_names:
                raise ValueError(f'Unknown argument {arg}')

        # Set field values
        for arg in field_names:
            setattr(self, arg, kwargs.get(arg, None))

    def export_proto(self):
        """Construct and return a protobuf message."""

        # Construct Protobuf message
        if base_has_export_proto:
            proto = base_class.export_proto(self)
            message = getattr(proto, base_field_name)
            message.SetInParent()
        else:
            # TODO (trb 08/01/2019): This list would have to be
            # updated any time another _pb2 file is created. It might
            # be better to have this as a global `frozenset`
            # (ndryden's suggestion) that gets maintained
            # elsewhere. But this code either works or doesn't get
            # executed now, so I vote delaying this fix until a need
            # arises.
            proto_modules = [callbacks_pb2, layers_pb2, metrics_pb2, model_pb2, objective_functions_pb2, operators_pb2, optimizers_pb2, training_algorithm_pb2, weights_pb2]
            proto_type = None
            while proto_type is None:
                proto_type = getattr(proto_modules.pop(), message_name, None)
            proto = proto_type()
            message = proto

        # Set message
        kwargs = {}
        for field_name in field_names:
            val = getattr(self, field_name, None)
            if val is not None:
                kwargs[field_name] = val
        set_protobuf_message(message, **kwargs)

        # Return Protobuf message
        return proto

    def get_field_names(self):
        """Names of parameters in derived class."""
        return field_names

    # Generate docstring
    if message_descriptor.fields:
        doc = 'Fields:\n'
        for field in message_descriptor.fields:
            doc += '    {0} ({1} {2})\n'.format(
                field.name,
                _protobuf_field_label_names.get(field.label, 'unknown'),
                _protobuf_field_type_names.get(field.type, 'unknown'))
    else:
        doc = 'Fields: none\n'

    # Create new class
    class_dictionary = {'__init__': __init__,
                        '__doc__': doc,
                        'export_proto': export_proto,
                        'get_field_names': get_field_names}
    class_dictionary.update(all_enums)

    return type(message_name, (base_class,), class_dictionary)

def generate_classes_from_protobuf_message(message,
                                           skip_fields = set(),
                                           base_class = object,
                                           base_kwargs = set(),
                                           base_has_export_proto = False):
    """Generate new classes based on fields in a Protobuf message.

    Args:
        message (type): A derived class of
            `google.protobuf.message.Message`. A new class will be
            generated for each field in the message.
        skip_fields (Iterable of str, optional): Protobuf message
            fields to ignore.
        base_class (type, optional): Generated classes will inherit
            from this class.
        base_kwargs (Iterable of str, optional): Keyword arguments for
            base class `__init__` method.
        base_has_export_proto (bool, optional): Whether the base class
            implements an `export_proto` method. If `True`, the base
            class `export_proto` is responsible for constructing a
            message of type `message` and the generated class
            `export_proto` will set the appropriate field.

    Returns:
        list of type: Generated classes.

    """
    classes = []
    for field in message.DESCRIPTOR.fields:
        if field.name not in skip_fields:
            classes.append(_generate_class(field.message_type,
                                           field.name,
                                           base_class,
                                           base_kwargs,
                                           base_has_export_proto))
    return classes

def set_protobuf_message(message, **kwargs):

    field_descriptors = message.DESCRIPTOR.fields_by_name

    # Iterate through kwargs
    for field_name, value in kwargs.items():

        # Make sure kwarg corresponds to field in message
        if field_name not in field_descriptors:
            raise KeyError(
                f'Protobuf message {message.DESCRIPTOR.name} '
                f'has no field "{field_name}"')

        # Do nothing if value is None
        if value is None:
            continue

        # Attempt to set field
        try:
            field = getattr(message, field_name)
            field_descriptor = field_descriptors[field_name]
            if field_descriptor.message_type in _protobuf_type_wrappers:
                field.SetInParent()
                field.value = value
            elif field_descriptor.label == google.protobuf.descriptor.FieldDescriptor.LABEL_REPEATED:
                try:
                    iterable_value = make_iterable(value)
                    if field_descriptor.type == field_descriptor.TYPE_MESSAGE:
                        field.extend([x.export_proto() for x in iterable_value])
                    else:
                        field.extend(iterable_value)
                except TypeError:
                    if (type(value).__module__ == 'numpy' and type(value).__name__ == 'nditer'):
                        value.reset()
                        for v in value:
                            field.append(float(v))
                    else:
                        raise
            elif isinstance(value, google.protobuf.message.Message):
                getattr(message, field_name).MergeFrom(value)
            elif callable(getattr(value, "export_proto", None)):
                # 'value' is (hopefully) an LBANN class
                # representation of a protobuf message.
                getattr(message, field_name).MergeFrom(value.export_proto())
            else:
                setattr(message, field_name, value)
        except:
            raise TypeError(
                f'Attempted to set field "{field_name}" in '
                f'Protobuf message {message.DESCRIPTOR.name} '
                f'with a {type(value).__name__}')

def get_parallel_strategy_args(**kwargs):
    """A wrapper function to create parallel_strategy arguments for
    Distconv-enabled layers.

    Args:
        {sample, depth, height, width, channel, filter}_groups (int):
            The number of processes for the corresponding dimension.
    """

    dimension_names = [
        "sample",
        "depth",
        "height",
        "width",
        "channel",
        "filter",
    ]
    group_names = ["{}_groups".format(x) for x in dimension_names]
    assert len(set(kwargs.keys())-set(group_names)) == 0

    parallel_strategy = {}
    for group_name in group_names:
        parallel_strategy[group_name] = kwargs[group_name] \
            if group_name in kwargs.keys() else 1

    return parallel_strategy

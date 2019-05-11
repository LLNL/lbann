"""Utility functions to generate classes from Protobuf messages."""
import google.protobuf.descriptor
from lbann import lbann_pb2

# Map from Protobuf label enums to strings
_proto_label_to_str = {
    google.protobuf.descriptor.FieldDescriptor.LABEL_OPTIONAL: 'optional',
    google.protobuf.descriptor.FieldDescriptor.LABEL_REQUIRED: 'required',
    google.protobuf.descriptor.FieldDescriptor.LABEL_REPEATED: 'repeated'
}
# Map from Protobuf type enums to strings
_proto_type_to_str = {
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
    field_names = message_descriptor.fields_by_name.keys()

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
                raise ValueError('Unknown argument {0}'.format(arg))

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
            proto = getattr(lbann_pb2, message_name)()
            message = proto

        # Set message
        for field in field_names:
            val = getattr(self, field)
            if val is not None:
                if type(val) is list:
                    getattr(message, field).extend(val)
                else:
                    setattr(message, field, val)

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
                _proto_label_to_str.get(field.label, 'unknown'),
                _proto_type_to_str.get(field.type, 'unknown'))
    else:
        doc = 'Fields: none\n'

    # Create new class
    return type(message_name, (base_class,),
                {'__init__': __init__,
                 '__doc__': doc,
                 'export_proto': export_proto,
                 'get_field_names': get_field_names})

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

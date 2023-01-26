"""Tensor shape object and helpers."""
from lbann import datatype_pb2

class Shape:
    """A class representing a multidimensional shape."""

    def __init__(self, dims=None):
        self._shape = datatype_pb2.Shape(dims=dims)

    @property
    def dims(self):
        return self._shape.dims

    @dims.setter
    def set_dims(self, new_dims):
        self._shape = datatype_pb2.Shape(dims=new_dims)

    def export_proto(self):
        return self._shape

    def __repr__(self):
        return 'Shape(dims=' + repr(self._shape.dims) + ')'


def make_shapes(list_of_dims, *other_dims):
    if len(other_dims) == 0:  # make_shapes(dims)
        return [Shape(dims) for dims in list_of_dims]
    else:  # make_shapes(shape1, shape2, ...)
        return [Shape(list_of_dims)] + [Shape(dims) for dims in other_dims]

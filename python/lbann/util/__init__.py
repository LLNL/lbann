import collections.abc

def make_iterable(obj):
    """Convert to an iterable object.

    Simply returns `obj` if it is alredy iterable. Otherwise returns a
    1-tuple containing `obj`. `str`s are treated as _not_ iterable.

    """
    if isinstance(obj, collections.abc.Iterable) and not isinstance(obj, str):
        return obj
    else:
        return (obj,)

def str_list(it, sep=' '):
    """Convert an iterable object to a string."""
    return sep.join(str(i) for i in make_iterable(it))

def make_nd_array(*dims):
    """Create a multi-dimensional array with given dimensions.

    The multi-dimensional array is a nested list initialized with
    `None`s.

    """
    if dims:
        head, *tail = dims
        return [make_nd_array(*tail) for _ in range(head)]
    else:
        return None

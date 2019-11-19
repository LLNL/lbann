from collections.abc import Iterable

def make_iterable(obj):
    """Convert to an iterable object.

    Simply returns `obj` if it is alredy iterable. Otherwise returns a
    1-tuple containing `obj`.

    """
    if isinstance(obj, Iterable) and not isinstance(obj, str):
        return obj
    else:
        return (obj,)

def str_list(it, sep=' '):
    """Convert an iterable object to a string."""
    return sep.join(str(i) for i in make_iterable(it))

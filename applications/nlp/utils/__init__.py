import collections.abc

def make_iterable(obj):
    """Convert to an iterable object.

    Simply returns `obj` if it is alredy iterable. Otherwise returns a
    1-tuple containing `obj`.

    """
    if isinstance(obj, collections.abc.Iterable) and not isinstance(obj, str):
        return obj
    else:
        return (obj,)

def str_list(it):
    """Convert an iterable object to a space-separated string."""
    return ' '.join([str(i) for i in make_iterable(it)])

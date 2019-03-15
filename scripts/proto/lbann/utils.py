from collections.abc import Iterable
from os.path import dirname, abspath

def make_iterable(obj):
    """Convert to an iterable object.

    Simply returns `obj` if it is alredy iterable. Otherwise returns a
    1-tuple containing `obj`.

    """
    if isinstance(obj, Iterable):
        return obj
    else:
        return (obj,)

def lbann_dir():
    """LBANN root directory."""
    return dirname(dirname(dirname(dirname(abspath(__file__)))))

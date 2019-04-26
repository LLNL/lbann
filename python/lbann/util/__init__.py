from collections.abc import Iterable
from os.path import abspath, dirname, isfile, join

def lbann_exe():
    """LBANN executable."""
    install_dir = dirname(dirname(dirname(dirname(__file__))))
    exe = join(install_dir, 'bin', 'lbann')
    if isfile(exe):
        return exe
    else:
        # LBANN has been built with `build_lbann_lc.sh`
        import lbann.contrib.lc.paths
        return lbann.contrib.lc.paths.lbann_exe()

def make_iterable(obj):
    """Convert to an iterable object.

    Simply returns `obj` if it is alredy iterable. Otherwise returns a
    1-tuple containing `obj`.

    """
    if isinstance(obj, Iterable) and not isinstance(obj, str):
        return obj
    else:
        return (obj,)

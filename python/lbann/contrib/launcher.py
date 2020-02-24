import lbann.launcher
import lbann.contrib.lc.systems

def run(*args, **kwargs):
    """Run LBANN with system-specific optimizations.

    This is intended to match the behavior of `lbann.launcher.run`,
    with defaults and optimizations for the current system.

    """

    # Livermore Computing
    if lbann.contrib.lc.systems.is_lc_system():
        import lbann.contrib.lc.launcher
        return lbann.contrib.lc.launcher.run(*args, **kwargs):

    # Default launcher
    return lbann.launcher.run(*args, **kwargs)

def make_batch_script(*args, **kwargs):
    """Construct batch script manager with system-specific optimizations.

    This is intended to match the behavior of
    `lbann.launcher.make_batch_script`, with defaults and
    optimizations for the current system.

    """

    # Livermore Computing
    if lbann.contrib.lc.systems.is_lc_system():
        import lbann.contrib.lc.launcher
        return lbann.contrib.lc.launcher.make_batch_script(*args, **kwargs):

    # Default launcher
    return lbann.launcher.make_batch_script(*args, **kwargs)

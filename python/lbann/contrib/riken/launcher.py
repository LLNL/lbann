import os
import shutil
from lbann.contrib.riken.systems import *
import lbann.launcher
from lbann.util import make_iterable

def run(*args, **kwargs):
    """Run LBANN with RIKEN-specific optimizations (deprecated).

    This is deprecated. Use `lbann.contrib.launcher.run` instead.

    """

    import warnings
    warnings.warn(
        'Using deprecated function `lbann.contrib.riken.launcher.run`. '
        'Use `lbann.contrib.launcher.run` instead.'
    )
    from ..launcher import run as _run
    _run(*args, **kwargs)

def make_batch_script(
    system=system(),
    procs_per_node=procs_per_node(),
    scheduler=scheduler(),
    launcher_args=[],
    environment={},
    preamble_cmds=[],
    *args,
    **kwargs,
):
    """Construct batch script manager with RIKEN-specific optimizations.

    This is a wrapper around `lbann.launcher.make_batch_script`, with
    defaults and optimizations for RIKEN systems. See that function for a
    full list of options.

    """

    # Create shallow copies of input arguments
    launcher_args = list(make_iterable(launcher_args))
    environment = environment.copy()

    kwargs['work_dir'] = lbann.launcher.make_timestamped_work_dir(procs_per_node=procs_per_node, **kwargs)
    work_dir = kwargs['work_dir']

    LBANN_BIN=shutil.which('lbann')
    '# Cache commonly-used files'
    preamble_cmds.append(f'llio_transfer --purge {LBANN_BIN}')
    preamble_cmds.append(f'llio_transfer --purge {work_dir}/experiment.prototext')
    preamble_cmds.append(f'llio_transfer {LBANN_BIN}')
    preamble_cmds.append(f'llio_transfer {work_dir}/experiment.prototext')

    # Helper function to configure environment variables
    # Note: User-provided values take precedence, followed by values
    # in the environment, followed by default values.
    def set_environment(key, default):
        if key not in environment:
            environment[key] = os.getenv(key, default)

    # Optimized thread affinity for Fugaku
    if system == 'fugaku':
        cores_per_proc = cores_per_node(system) // procs_per_node
        set_environment('OMP_THREAD_LIMIT', cores_per_proc)
        set_environment('OMP_NUM_THREADS', cores_per_proc - 1)
        set_environment('LBANN_NUM_IO_THREADS', 1)
        set_environment('OMP_BIND', 'close')
        set_environment('LD_PRELOAD', '/usr/lib64/libhwloc.so.15')
        set_environment('LD_LIBRARY_PATH', '$(/home/system/tool/sort_libp)')

    return lbann.launcher.make_batch_script(
        procs_per_node=procs_per_node,
        scheduler=scheduler,
        launcher_args=launcher_args,
        environment=environment,
        preamble_cmds=preamble_cmds,
        *args,
        **kwargs,
    )

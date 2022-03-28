import os
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

    # Helper function to configure environment variables
    # Note: User-provided values take precedence, followed by values
    # in the environment, followed by default values.
    def set_environment(key, default):
        if key not in environment:
            environment[key] = os.getenv(key, default)

    # Setup GPU bindings
    # Note: Each Hydrogen process is assigned to the GPU index that
    # matches its node communicator rank. This is not compatible with
    # mpibind, which assigns a GPU with index 0 to each process. We
    # can't use an exclusive GPU compute mode since processes may
    # touch the wrong GPU while figuring out ownership.
    # if scheduler == 'pjm':
    #     launcher_args.extend(['OMP_SCHEDULE=static',
    #                           'OMP_BIND=close',
    #                           'FLIB_FASTOMP=FALSE OMP_NESTED=TRUE',
    #                           'LD_PRELOAD=/usr/lib64/libhwloc.so.15'])

                              # export OMP_WAIT_POLICY=ACTIVE
#'GOMP_SPINCOUNT=0',
        
    # Optimized thread affinity for Pascal
    # Note: Both GPUs are on socket 0, so we only use cores on that
    # socket.
    if system == 'fugaku':
        cores_per_proc = cores_per_node(system) // procs_per_node
#        set_environment('AL_PROGRESS_RANKS_PER_NUMA_NODE', procs_per_node)
        set_environment('OMP_THREAD_LIMIT', cores_per_proc)
        set_environment('OMP_NUM_THREADS', cores_per_proc - 1)
        set_environment('LBANN_NUM_IO_THREADS', 1)
#        set_environment('OMP_SCHEDULE', 'static')
#        set_environment('OMP_BIND', 'close')
#        set_environment('OMP_NESTED', 'TRUE')
#        set_environment('FLIB_FASTOMP', 'FALSE')
        set_environment('LD_PRELOAD', '/usr/lib64/libhwloc.so.15:libtcmalloc.so')
         

    return lbann.launcher.make_batch_script(
        procs_per_node=procs_per_node,
        scheduler=scheduler,
        launcher_args=launcher_args,
        environment=environment,
        *args,
        **kwargs,
    )

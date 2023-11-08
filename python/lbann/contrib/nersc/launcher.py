import math
import os
from lbann.contrib.nersc.systems import *
import lbann.launcher
from lbann.util import make_iterable

def make_batch_script(
    system=system(),
    procs_per_node=procs_per_node(),
    scheduler=scheduler(),
    launcher_args=[],
    environment={},
    *args,
    **kwargs,
):
    """Construct batch script manager with NERSC-specific optimizations.

    This is a wrapper around `lbann.launcher.make_batch_script`, with
    defaults and optimizations for NERSC systems. See that function for a
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

    # Optimizations for Cori GPU nodes
    if system == 'cgpu':
        cores_per_proc = cores_per_node(system) // procs_per_node
        set_environment('AL_PROGRESS_RANKS_PER_NUMA_NODE',
                        math.ceil(procs_per_node / numa_nodes_per_node(system)))
        if scheduler == 'slurm':
            masks = [2**cores_per_proc - 1]
            while len(masks) < procs_per_node:
                masks.append(masks[-1] << cores_per_proc)
            mask_str = ','.join([hex(mask) for mask in masks])
            launcher_args.append('--cpu_bind=mask_cpu:{}'.format(mask_str))

        launcher_args.extend(['--qos=regular',
                              f'--cpus-per-task={cores_per_proc}',
                              '--gpus-per-task=1',
                              '--constraint=gpu'])

        # Hack to enable process forking
        # Note: InfiniBand is known to experience hangs if an MPI
        # process is forked (see
        # https://www.open-mpi.org/faq/?category=openfabrics#ofa-fork).
        # Setting IBV_FORK_SAFE seems to fix this issue, but it may
        # hurt performance (see
        # https://linux.die.net/man/3/ibv_fork_init).
        set_environment('IBV_FORK_SAFE', 1)

        set_environment('MV2_ENABLE_AFFINITY', 0)

        set_environment('MV2_USE_CUDA', 1)

        set_environment('MKL_THREADING_LAYER', 'GNU')

    return lbann.launcher.make_batch_script(
        procs_per_node=procs_per_node,
        scheduler=scheduler,
        launcher_args=launcher_args,
        environment=environment,
        *args,
        **kwargs,
    )

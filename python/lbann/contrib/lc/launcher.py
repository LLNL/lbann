from lbann import lbann_exe
from lbann.contrib.lc.systems import *
import lbann.launcher

def run(model, data_reader, optimizer,
        lbann_exe=lbann_exe(),
        lbann_args='',
        experiment_dir=None,
        nodes=1,
        procs_per_node=procs_per_node(),
        time_limit=60,
        scheduler=scheduler(),
        job_name='lbann',
        system=system(),
        partition=partition(),
        account=account(),
        reservation=None,
        launcher_args='',
        environment={},
        setup_only=False):
    """Run LBANN experiment with LC-specific optimizations.

    This is a convenience wrapper around the `lbann.launcher.run`
    function, with defaults and optimizations for LC systems.

    """

    # Setup GPU bindings
    # Note: Hydrogen processes take ownership of the GPU indices that
    # matches their node communicator ranks. mpibind assigns each rank
    # a unique GPU with index 0, so it should be disabled. Processes
    # may touch the wrong GPUs in the process of figuring out GPU
    # ownership, so an exclusive GPU compute mode causes problems.
    if scheduler == 'slurm' and has_gpu(system):
        launcher_args += ' --mpibind=off --nvidia_compute_mode=default'

    # Deal with Pascal's strange hardware topology
    # Note: Both GPUs on a Pascal node are on the same socket, so we
    # only use cores on that socket.
    if system == 'pascal' and procs_per_node == 2:
        if scheduler == 'slurm':
            launcher_args += ' --cpu_bind=mask_cpu:0x000001ff,0x0003fe00'
        environment['OMP_NUM_THREADS'] = 8
        environment['AL_PROGRESS_RANKS_PER_NUMA_NODE'] = 2

    # Hacked bugfix for MPI_Init in MVAPICH2-2.3
    # Note: MPI_Init hangs when started with more than 35
    # processes. This bug is not present in MVAPICH2-2.2 but is
    # present in MVAPICH2-2.3rc2.
    environment['MV2_USE_RDMA_CM'] = 0

    # Hacked bugfix for MPI_Sendrecv in MVAPICH2-2.3
    # Note: MPI_Sendrecv produces incorrect output under certain
    # circumstances. This bug is not present in MVAPICH2-2.2 or
    # MVAPICH2-2.3.1.
    environment['MV2_USE_LAZY_MEM_UNREGISTER'] = 0

    # Magic default arguments to jsrun/etc.
    # Note: Pack processes using ten cores for each, with 40 cores total, and
    # all four GPUs visible to each process.
    if system in ('sierra', 'lassen'):
        if scheduler == 'lsf':
            launcher_args += ' -d packed -b "packed:10" -r 1 -c 40 -g 4'
        environment['OMP_NUM_THREADS'] = 4
        # Deal with topology mis-identification on Sierra/Lassen.
        environment['AL_PROGRESS_RANKS_PER_NUMA_NODE'] = 2

    # Run LBANN
    return lbann.launcher.run(model, data_reader, optimizer,
                              lbann_exe=lbann_exe,
                              lbann_args=lbann_args,
                              experiment_dir=experiment_dir,
                              nodes=nodes,
                              procs_per_node=procs_per_node,
                              time_limit=time_limit,
                              scheduler=scheduler,
                              job_name=job_name,
                              system=system,
                              partition=partition,
                              account=account,
                              reservation=reservation,
                              launcher_args=launcher_args,
                              environment=environment,
                              setup_only=setup_only)

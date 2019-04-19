from lbann.contrib.lc.systems import *
from lbann.contrib.lc.paths import lbann_exe
import lbann.launcher

def run(model, data_reader, optimizer,
        lbann_exe = lbann_exe(),
        lbann_args = '',
        experiment_dir = None,
        nodes = 1,
        procs_per_node = procs_per_node(),
        time_limit = 60,
        scheduler = scheduler(),
        job_name = 'lbann',
        system = system(),
        partition = partition(),
        account = account(),
        launcher_args = '',
        environment = {},
        setup_only = False):
    """Run LBANN experiment with LC-specific optimizations.

    This is a convenience wrapper around the `lbann.launcher.run`
    function, with defaults and optimizations for LC systems.

    """

    # Hacked bugfix for MVAPICH
    # TODO: Is this still needed?
    environment['MV2_USE_RDMA_CM'] = 0

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

    # Magic default arguments to jsrun/etc.
    if system in ('sierra', 'lassen'):
        if scheduler == 'lsf':
            launcher_args += ' -d packed -b "packed:10" -r 1 -c 40 -g 4'
        environment['OMP_NUM_THREADS'] = 4
        # Deal with topology mis-identification on Sierra/Lassen.
        environment['AL_PROGRESS_RANKS_PER_NUMA_NODE'] = 2

    # Run LBANN
    lbann.launcher.run(lbann_exe, model, data_reader, optimizer,
                       lbann_args = lbann_args,
                       experiment_dir = experiment_dir,
                       nodes = nodes,
                       procs_per_node = procs_per_node,
                       time_limit = time_limit,
                       scheduler = scheduler,
                       job_name = job_name,
                       system = system,
                       partition = partition,
                       account = account,
                       launcher_args = launcher_args,
                       environment = environment,
                       setup_only = setup_only)

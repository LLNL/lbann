import os, os.path
from lbann import lbann_exe
from lbann.contrib.lc.systems import *
import lbann.launcher
from lbann.util import make_iterable

def run(trainer, model, data_reader, optimizer,
        experiment_dir=None,
        nodes=1,
        procs_per_node=procs_per_node(),
        time_limit=None,
        scheduler=scheduler(),
        job_name='lbann',
        system=system(),
        partition=partition(),
        account=account(),
        reservation=None,
        overwrite_script=False,
        launcher_args=[],
        lbann_args=[],
        environment={},
        setup_only=False):
    """Run LBANN with LC-specific optimizations.

    This is intended to match the behavior of `lbann.launcher.run`,
    with defaults and optimizations for LC systems.

    """

    # Create batch script generator
    script = make_batch_script(work_dir=experiment_dir,
                               nodes=nodes,
                               procs_per_node=procs_per_node,
                               time_limit=time_limit,
                               scheduler=scheduler,
                               job_name=job_name,
                               partition=partition,
                               account=account,
                               reservation=reservation,
                               launcher_args=launcher_args,
                               environment=environment)

    # Check for an existing job allocation
    has_allocation = False
    if isinstance(script, lbann.launcher.slurm.SlurmBatchScript):
        has_allocation = 'SLURM_JOB_ID' in os.environ
    if isinstance(script, lbann.launcher.lsf.LSFBatchScript):
        has_allocation = 'LSB_JOBID' in os.environ

    # Batch script prints start time
    script.add_command('echo "Started at $(date)"')

    # Batch script invokes LBANN
    lbann_command = [lbann.lbann_exe()]
    lbann_command.extend(make_iterable(lbann_args))
    prototext_file = os.path.join(script.work_dir, 'experiment.prototext')
    lbann.proto.save_prototext(prototext_file,
                               trainer=trainer,
                               model=model,
                               data_reader=data_reader,
                               optimizer=optimizer)
    lbann_command.append('--prototext={}'.format(prototext_file))
    script.add_parallel_command(lbann_command)
    script.add_command('status=$?')

    # Batch script prints finish time and returns status
    script.add_command('echo "Finished at $(date)"')
    script.add_command('exit ${status}')

    # Write, run, or submit batch script
    status = 0
    if setup_only:
        script.write(overwrite=overwrite_script)
    elif has_allocation:
        status = script.run(overwrite=overwrite_script)
    else:
        status = script.submit(overwrite=overwrite_script)
    return status

def make_batch_script(script_file=None,
                      work_dir=None,
                      nodes=1,
                      procs_per_node=procs_per_node(),
                      time_limit=None,
                      scheduler=scheduler(),
                      job_name='lbann',
                      system=system(),
                      partition=partition(),
                      account=account(),
                      reservation=None,
                      launcher_args=[],
                      environment={}):
    """Construct batch script manager with LC-specific optimizations.

    This is intended to match the behavior of
    `lbann.launcher.make_batch_script`, with defaults and
    optimizations for LC systems.

    """

    # Create shallow copies of input arguments
    launcher_args = list(make_iterable(launcher_args))
    environment = environment.copy()

    # Setup GPU bindings
    # Note: Each Hydrogen process is assigned to the GPU index that
    # matches its node communicator rank. This is not compatible with
    # mpibind, which assigns a GPU with index 0 to each process. We
    # can't use an exclusive GPU compute mode since processes may
    # touch the wrong GPU while figuring out ownership.
    if scheduler == 'slurm' and has_gpu(system):
        launcher_args.extend(['--mpibind=off',
                              '--nvidia_compute_mode=default'])

    # Optimized thread affinity for Pascal
    # Note: Both GPUs are on socket 0, so we only use cores on that
    # socket.
    if system == 'pascal':
        cores_per_socket = cores_per_node(system) // 2
        cores_per_proc = cores_per_socket // procs_per_node
        if 'AL_PROGRESS_RANKS_PER_NUMA_NODE' not in environment:
            environment['AL_PROGRESS_RANKS_PER_NUMA_NODE'] = procs_per_node
        if 'OMP_NUM_THREADS' not in environment:
            environment['OMP_NUM_THREADS'] = cores_per_proc - 1
        if scheduler == 'slurm':
            masks = [2**cores_per_proc - 1]
            while len(masks) < procs_per_node:
                masks.append(masks[-1] << cores_per_proc)
            mask_str = ','.join([hex(mask) for mask in masks])
            launcher_args.append('--cpu_bind=mask_cpu:{}'.format(mask_str))

    # Hacked bugfix for MPI_Init in MVAPICH2-2.3 (8/23/18)
    # Note: MPI_Init hangs when started with more than 35
    # processes. This bug is not present in MVAPICH2-2.2 but is
    # present in MVAPICH2-2.3rc2.
    if 'MV2_USE_RDMA_CM' not in environment:
        environment['MV2_USE_RDMA_CM'] = 0

    # Optimizations for Sierra-like systems
    if system in ('sierra', 'lassen'):

        # Set thread affinity
        # Note: Aluminum's default thread affinity is incorrect since
        # hwloc treats GPUs as NUMA domains.
        # Note: There are actually 22 cores/socket, but it seems that
        # powers of 2 are better for performance.
        cores_per_socket = 16
        procs_per_socket = (procs_per_node + 1) // 2
        cores_per_proc = cores_per_socket // procs_per_socket
        if 'AL_PROGRESS_RANKS_PER_NUMA_NODE' not in environment:
            environment['AL_PROGRESS_RANKS_PER_NUMA_NODE'] = procs_per_socket
        if 'OMP_NUM_THREADS' not in environment:
            environment['OMP_NUM_THREADS'] = cores_per_proc
        if scheduler == 'lsf':
            launcher_args.append('--bind packed:{}'.format(cores_per_proc))

        # Hack to enable process forking
        # Note: InfiniBand is known to experience hangs if an MPI
        # process is forked (see
        # https://www.open-mpi.org/faq/?category=openfabrics#ofa-fork).
        # Setting IBV_FORK_SAFE seems to fix this issue, but it may
        # hurt performance (see
        # https://linux.die.net/man/3/ibv_fork_init).
        if 'IBV_FORK_SAFE' not in environment:
            environment['IBV_FORK_SAFE'] = 1

        # Hacked bugfix for hcoll (1/23/19)
        # Note: Fixes hangs in MPI_Bcast.
        if 'HCOLL_ENABLE_SHARP' not in environment:
            environment['HCOLL_ENABLE_SHARP'] = 0
        if 'OMPI_MCA_coll_hcoll_enable' not in environment:
            environment['OMPI_MCA_coll_hcoll_enable'] = 0

        # Hacked bugfix for Spectrum MPI PAMI (9/17/19)
        if 'PAMI_MAX_NUM_CACHED_PAGES' not in environment:
            environment['PAMI_MAX_NUM_CACHED_PAGES'] = 0

    return lbann.launcher.make_batch_script(script_file=script_file,
                                            work_dir=work_dir,
                                            nodes=nodes,
                                            procs_per_node=procs_per_node,
                                            time_limit=time_limit,
                                            scheduler=scheduler,
                                            job_name=job_name,
                                            partition=partition,
                                            account=account,
                                            reservation=reservation,
                                            launcher_args=launcher_args,
                                            environment=environment)

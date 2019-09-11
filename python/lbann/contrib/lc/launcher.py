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
        script.write()
    elif has_allocation:
        status = script.run()
    else:
        status = script.submit()
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
    # Note: Hydrogen processes take ownership of the GPU indices that
    # matches their node communicator ranks. mpibind assigns each rank
    # a unique GPU with index 0, so it should be disabled. Processes
    # may touch the wrong GPUs in the process of figuring out GPU
    # ownership, so an exclusive GPU compute mode causes problems.
    if scheduler == 'slurm' and has_gpu(system):
        launcher_args.extend(['--mpibind=off',
                              '--nvidia_compute_mode=default'])

    # Deal with Pascal's hardware topology
    # Note: Both GPUs on a Pascal node are on the same socket, so we
    # only use cores on that socket.
    if system == 'pascal' and procs_per_node == 2:
        if scheduler == 'slurm':
            launcher_args.append('--cpu_bind=mask_cpu:0x000001ff,0x0003fe00')
        environment['OMP_NUM_THREADS'] = 8
        environment['AL_PROGRESS_RANKS_PER_NUMA_NODE'] = 2

    # Hacked bugfix for MPI_Init in MVAPICH2-2.3
    # Note: MPI_Init hangs when started with more than 35
    # processes. This bug is not present in MVAPICH2-2.2 but is
    # present in MVAPICH2-2.3rc2.
    environment['MV2_USE_RDMA_CM'] = 0

    # Magic default arguments to jsrun/etc.
    # Note: Pack processes using ten cores for each, with 40 cores total, and
    # all four GPUs visible to each process.
    if system in ('sierra', 'lassen'):
        if scheduler == 'lsf':
            launcher_args.extend([
                '--launch_distribution packed',
                '--bind "packed:10"',
                '--rs_per_host 1',
                '--cpu_per_rs 40',
                '--gpu_per_rs 4'
            ])
        environment['OMP_NUM_THREADS'] = 4
        # Deal with topology mis-identification on Sierra/Lassen.
        environment['AL_PROGRESS_RANKS_PER_NUMA_NODE'] = 2

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

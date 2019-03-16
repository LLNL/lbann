"""Utility functions for Slurm on LC systems."""
import os
import os.path
import subprocess
from lbann.lc.systems import (system, partition, account, has_gpu,
                              procs_per_node, time_limit)
from lbann.lc.paths import lbann_exe

def run(experiment_dir = os.getcwd(),
        exe = lbann_exe(),
        exe_args = '',
        srun_args = '',
        job_name = 'lbann',
        nodes = 1,
        procs_per_node = procs_per_node(),
        system = system(),
        partition = partition(),
        account = account(),
        time_limit = time_limit(),
        setup_only = False):
    """Run executable with Slurm.

    Creates a Slurm batch script in the experiment directory. If a
    Slurm job allocation is detected, the script is run
    directly. Otherwise, the script is submitted to sbatch.

    Args:
        experiment_dir (str, optional): Experiment directory.
        exe (str, optional): Executable.
        exe_args (str, optional): Command-line arguments to executable.
        srun_args (str, optional): Command-line arguments to srun.
        job_name (str, optional): Batch job name.
        nodes (int, optional): Number of compute nodes.
        procs_per_node (int, optional): Number of processes per compute
            node.
        system (str, optional): Target system.
        partition (str, optional): Scheduler partition.
        account (str, optional): Scheduler account.
        time_limit (int, optional): Job time limit, in minutes.
        setup_only (bool, optional): If true, the experiment is not
            run after the batch script is created.

    """

    # Check for an existing job allocation from Slurm
    # Note: Settings for current job allocation take precedence
    has_allocation = 'SLURM_JOB_ID' in os.environ
    if has_allocation:
        job_name = os.environ['SLURM_JOB_NAME']
        partition = os.environ['SLURM_JOB_PARTITION']
        account = os.environ['SLURM_JOB_ACCOUNT']
        time_limit = -1

    # Experiment directory
    experiment_dir = os.path.abspath(experiment_dir)
    os.makedirs(experiment_dir, exist_ok=True)
    batch_file = os.path.join(experiment_dir, 'batch.sh')
    out_file = os.path.join(experiment_dir, 'out.log')
    err_file = os.path.join(experiment_dir, 'err.log')
    nodes_file = os.path.join(experiment_dir, 'nodes.txt')

    # srun command-line arguments
    if has_gpu(system):
        srun_args += ' --mpibind=off --nvidia_compute_mode=default'
    if system == 'pascal' and procs_per_node == 2:
        srun_args += ' --cpu_bind=mask_cpu:0x000001ff,0x0003fe00'

    # Write batch script
    with open(batch_file, 'w') as f:
        f.write('#!/bin/sh\n')

        # Slurm job settings
        if job_name:
            f.write('#SBATCH --job-name={}\n'.format(job_name))
        f.write('#SBATCH --nodes={}\n'.format(nodes))
        if partition:
            f.write('#SBATCH --partition={}\n'.format(partition))
        if account:
            f.write('#SBATCH --account={}\n'.format(account))
        f.write('#SBATCH --workdir={}\n'.format(experiment_dir))
        f.write('#SBATCH --output={}\n'.format(out_file))
        f.write('#SBATCH --error={}\n'.format(err_file))
        if time_limit >= 0:
            hours, minutes = divmod(int(time_limit), 60)
            days, hours = divmod(hours, 24)
            f.write('#SBATCH --time={}-{:02d}:{:02d}:00\n'
                    .format(days, hours, minutes))
        f.write('\n')

        # Set environment
        f.write('# ==== Environment ====\n')
        f.write('export MV2_USE_RDMA_CM=0\n')
        if system == 'pascal':
            f.write('export OMP_NUM_THREADS=8\n')
            f.write('export AL_PROGRESS_RANKS_PER_NUMA_NODE=2\n')
        f.write('\n')

        # Display time and node list
        f.write('# ==== Useful info ====\n')
        f.write('date\n')
        f.write('srun --nodes={0} --ntasks={0} hostname > {1}\n'
                .format(nodes, nodes_file))
        f.write('sort --unique --output={0} {0}\n'.format(nodes_file))
        f.write('\n')

        # Run experiment
        f.write('# ==== Experiment ====\n')
        f.write('srun {} --nodes={} --ntasks={} {} {}\n'
                .format(srun_args, nodes, nodes * procs_per_node,
                        exe, exe_args))

    # Make batch script executable
    os.chmod(batch_file, 0o755)

    # Launch job if needed
    # Note: Pipes output to log files
    if not setup_only:
        run_exe = 'sh' if has_allocation else 'sbatch'
        run_proc = subprocess.Popen([run_exe, batch_file],
                                    stdout = subprocess.PIPE,
                                    stderr = subprocess.PIPE,
                                    cwd = experiment_dir)
        out_proc = subprocess.Popen(['tee', out_file],
                                    stdin = run_proc.stdout,
                                    cwd = experiment_dir)
        err_proc = subprocess.Popen(['tee', err_file],
                                    stdin = run_proc.stderr,
                                    cwd = experiment_dir)
        run_proc.stdout.close()
        run_proc.stderr.close()
        run_proc.wait()
        out_proc.wait()
        err_proc.wait()

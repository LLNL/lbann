"""Utility functions for Slurm on LC systems."""
import os
import os.path
import subprocess
from lbann.lc.systems import system, partition, account, time_limit, procs_per_node
from lbann.lc.paths import lbann_exe

def run(work_dir = os.getcwd(),
        lbann_exe = lbann_exe(),
        lbann_args = '',
        prototext_file = 'experiment.prototext',
        srun_args = '',
        job_name = 'lbann',
        nodes = 1,
        procs_per_node = procs_per_node(),
        partition = partition(),
        account = account(),
        time_limit = time_limit()):
    """Run LBANN experiment with Slurm.

    Creates a Slurm batch script in the work directory. If a Slurm job
    allocation is detected, the script is run directly. Otherwise, the
    script is submitted to sbatch.

    Args:
        work_dir (str, optional): Work directory.
        lbann_exe (str, optional): LBANN executable.
        lbann_args (str, optional): Command-line arguments to LBANN
            executable.
        prototext_file (str, optional): Prototext file for LBANN
            experiment.
        srun_args (str, optional): Command-line arguments to srun.
        job_name (str, optional): Batch job name.
        nodes (int, optional): Number of compute nodes.
        procs_per_node (int, optional): Number of processes per compute
            node.
        partition (str, optional): Scheduler partition.
        account (str, optional): Scheduler account.
        time_limit (int, optional): Job time limit, in minutes.

    """

    # Determine if nodes have already been allocated by Slurm
    has_allocation = 'SLURM_JOB_ID' in os.environ
    if has_allocation:
        job_name = os.environ['SLURM_JOB_NAME']
        partition = os.environ['SLURM_JOB_PARTITION']
        account = os.environ['SLURM_JOB_ACCOUNT']
        time_limit = -1

    # Initialize experiment directory
    batch_file, out_file, err_file = setup(
        prototext_file = prototext_file,
        work_dir = work_dir,
        job_name = job_name,
        lbann_exe = lbann_exe,
        lbann_args = lbann_args,
        nodes = nodes,
        procs_per_node = procs_per_node,
        system = system(),
        partition = partition,
        account = account,
        time_limit = time_limit)

    # Launch job
    # Note: Pipes output to log files
    run_exe = 'sh' if has_allocation else 'sbatch'
    run_proc = subprocess.Popen([run_exe, batch_file],
                                stdout = subprocess.PIPE,
                                stderr = subprocess.PIPE,
                                cwd = work_dir)
    out_proc = subprocess.Popen(['tee', out_file],
                                stdin = run_proc.stdout,
                                cwd = work_dir)
    err_proc = subprocess.Popen(['tee', err_file],
                                stdin = run_proc.stderr,
                                cwd = work_dir)
    run_proc.stdout.close()
    run_proc.stderr.close()
    run_proc.wait()
    out_proc.wait()
    err_proc.wait()

def setup(work_dir = os.getcwd(),
          lbann_exe = lbann_exe(),
          lbann_args = '',
          prototext_file = '',
          srun_args = '',
          job_name = 'lbann',
          nodes = 1,
          procs_per_node = procs_per_node(),
          system = system(),
          partition = partition(),
          account = account(),
          time_limit = time_limit()):
    """Create Slurm batch script in work directory.

    Args:
        work_dir (str, optional): Work directory.
        lbann_exe (str, optional): LBANN executable.
        lbann_args (str, optional): Command-line arguments to LBANN
            executable.
        prototext_file (str, optional): Prototext file for LBANN
            experiment.
        lbann_args (str, optional): Command-line arguments to srun.
        job_name (str, optional): Batch job name.
        nodes (int, optional): Number of compute nodes.
        procs_per_node (int, optional): Number of processes per compute
            node.
        system (str, optional): Target system for batch job.
        partition (str, optional): Scheduler partition.
        account (str, optional): Scheduler account.
        time_limit (int, optional): Job time limit, in minutes.

    Returns:
        batch_file (str): Batch script.
        out_file (str): Log file for stdout.
        err_file (str): Log file for stderr.

    """

    # Create work directory if needed
    work_dir = os.path.abspath(work_dir)
    os.makedirs(work_dir, exist_ok=True)

    # Batch script and generated files
    batch_file = os.path.join(work_dir, 'batch.sh')
    out_file = os.path.join(work_dir, 'out.log')
    err_file = os.path.join(work_dir, 'err.log')
    nodes_file = os.path.join(work_dir, 'nodes.txt')

    # srun command-line arguments
    if system in ('pascal', 'surface'):
        srun_args += ' --mpibind=off --nvidia_compute_mode=default'
    if system == 'pascal':
        srun_args += ' --cpu_bind=mask_cpu:0x000001ff,0x0003fe00'

    # LBANN command-line arguments
    if prototext_file:
        lbann_args += ' --prototext=' + prototext_file

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
        f.write('#SBATCH --workdir={}\n'.format(work_dir))
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
                        lbann_exe, lbann_args))

    # Make batch script executable
    os.chmod(batch_file, 0o755)

    # Return batch script path
    return batch_file, out_file, err_file

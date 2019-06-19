"""Utility functions for Slurm."""
import os, os.path
import subprocess
from lbann.util import make_iterable

def run(command,
        experiment_dir=os.getcwd(),
        nodes=1,
        procs_per_node=1,
        time_limit=-1,
        job_name=None,
        partition=None,
        account=None,
        reservation=None,
        srun_args='',
        environment={},
        setup_only=False):
    """Run executable with Slurm.

    Creates a Slurm batch script in the experiment directory. If a
    Slurm job allocation is detected, the script is run
    directly. Otherwise, the script is submitted to sbatch.

    Args:
        command (str): Program to run under Slurm, i.e. an executable
            and its command-line arguments.
        experiment_dir (str, optional): Experiment directory.
        nodes (int, optional): Number of compute nodes.
        procs_per_node (int, optional): Number of processes per compute
            node.
        time_limit (int, optional): Job time limit, in minutes. A
            negative value implies the system-default time limit.
        job_name (str, optional): Batch job name.
        partition (str, optional): Scheduler partition.
        account (str, optional): Scheduler account.
        reservation (str, optional): Scheduler reservation name.
        srun_args (str, optional): Command-line arguments to srun.
        environment (dict of {str: str}, optional): Environment
            variables.
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
        if reservation:
            raise ValueError('Slurm reservations not supported')
        f.write('#SBATCH --workdir={}\n'.format(experiment_dir))
        f.write('#SBATCH --output={}\n'.format(out_file))
        f.write('#SBATCH --error={}\n'.format(err_file))
        if time_limit >= 0:
            seconds = int((time_limit % 1) * 60)
            hours, minutes = divmod(int(time_limit), 60)
            days, hours = divmod(hours, 24)
            f.write('#SBATCH --time={}-{:02d}:{:02d}:{:02d}\n'
                    .format(days, hours, minutes, seconds))

        # Set environment
        if environment:
            f.write('\n')
            f.write('# ==== Environment ====\n')
            for variable, value in environment.items():
                f.write('export {}={}\n'.format(variable, value))

        # Display time and node list
        f.write('\n')
        f.write('# ==== Useful info ====\n')
        f.write('date\n')
        f.write('srun --nodes={0} --ntasks={0} hostname > {1}\n'
                .format(nodes, nodes_file))
        f.write('sort --unique --output={0} {0}\n'.format(nodes_file))

        # Run experiment
        f.write('\n')
        f.write('# ==== Experiment ====\n')
        for cmd in make_iterable(command):
            f.write('srun {} --nodes={} --ntasks={} {}\n'
                    .format(srun_args, nodes, nodes * procs_per_node,
                            cmd))

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

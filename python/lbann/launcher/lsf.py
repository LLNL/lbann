"""Utility functions for LSF."""

import os
import os.path
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
        jsrun_args='',
        environment={},
        setup_only=False):
    """Run executable with LSF.

    Creates an LSF batch script in the experiment directory. If a LSF
    job allocation is detected, the script is run directly. Otherwise,
    the script is submitted to bsub.

    Args:
        command (str): Program to run under LSF, i.e. an executable and
            its command-line arguments.
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
        jsrun_args (str, optional): Command-line arguments to jsrun.
        environment (dict of {str: str}, optional): Environment
            variables.
        setup_only (bool, optional): If true, the experiment is not
            run after the batch script is created.

    Returns:
        int: Exit status from LSF. This is really only meaningful if
            the script is run on an existing node allocation. If a
            batch job is submitted, LSF will probably return 0
            trivially.

    """
    # Check for an existing job allocation.
    # Note: Settings for existing allocations take precedence.
    has_allocation = 'LSB_JOBID' in os.environ
    if has_allocation:
        job_name = os.environ['LSB_JOBNAME']
        partition = os.environ['LSB_QUEUE']
        # LSF does not provide a way to get the account via env vars.
        time_limit = -1

    # Experiment directory
    experiment_dir = os.path.abspath(experiment_dir)
    os.makedirs(experiment_dir, exist_ok=True)
    batch_file = os.path.join(experiment_dir, 'batch.sh')
    out_file = os.path.join(experiment_dir, 'out.log')
    err_file = os.path.join(experiment_dir, 'err.log')
    nodes_file = os.path.join(experiment_dir, 'nodes.txt')

    # Create batch script.
    s = '#!/bin/sh\n'
    if job_name:
        s += '#BSUB -J {}\n'.format(job_name)
    s += '#BSUB -nnodes {}\n'.format(nodes)
    if partition:
        s += '#BSUB -q {}\n'.format(partition)
    if account:
        s += '#BSUB -G {}\n'.format(account)
    else:
        raise ValueError('LSF requires an account')
    if reservation:
        s += '#BSUB -U {}\n'.format(reservation)
    s += '#BSUB -cwd {}\n'.format(experiment_dir)
    s += '#BSUB -o {}\n'.format(out_file)
    s += '#BSUB -e {}\n'.format(err_file)
    if time_limit >= 0:
        s += '#BSUB -W {}\n'.format(time_limit)

    # Set environment variables.
    if environment:
        s += '\n# ==== Environment ====\n'
        for variable, value in environment.items():
            s += 'export {}={}\n'.format(variable, value)

    # Time and node list.
    s += '\n# ==== Useful info ====\n'
    s += 'date\n'
    s += 'jsrun -n {} -a 1 -r 1 hostname > {}\n'.format(nodes, nodes_file)
    s += 'sort --unique --output={0} {0}\n'.format(nodes_file)

    # Run experiment.
    s += '\n# ==== Experiment ====\n'
    for cmd in make_iterable(command):
        s += 'jsrun -n {} -a {} {} {}\n'.format(
            nodes, procs_per_node, jsrun_args, cmd)

    with open(batch_file, 'w') as f:
        f.write(s)

    # Make batch script executable.
    os.chmod(batch_file, 0o755)

    # Launch if needed.
    if setup_only:
        return 0
    else:
        if has_allocation:
            run_proc = subprocess.Popen(['sh', batch_file],
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE,
                                        cwd=experiment_dir)
        else:
            # bsub requires the batch script be read from its stdin.
            run_proc = subprocess.Popen('bsub < {}'.format(batch_file),
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE,
                                        cwd=experiment_dir,
                                        shell=True)
        out_proc = subprocess.Popen(['tee', out_file],
                                    stdin=run_proc.stdout,
                                    cwd=experiment_dir)
        err_proc = subprocess.Popen(['tee', err_file],
                                    stdin=run_proc.stderr,
                                    cwd=experiment_dir)
        run_proc.stdout.close()
        run_proc.stderr.close()
        run_proc.wait()
        out_proc.wait()
        err_proc.wait()
        return run_proc.returncode

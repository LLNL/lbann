"""Utility functions for LSF."""

import os
import os.path
import subprocess
from .batch_script import BatchScript
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

    # Initialize Slurm batch script
    script_file = os.path.join(experiment_dir, 'batch.sh')
    script = LSFBatchScript(script_file=script_file,
                            work_dir=experiment_dir,
                            nodes=nodes,
                            procs_per_node=procs_per_node,
                            time_limit=time_limit,
                            job_name=job_name,
                            partition=partition,
                            account=account,
                            reservation=reservation,
                            launcher_args=jsrun_args)

    # Set environment
    for variable, value in environment.items():
        script.add_command('export {0}={1}'.format(variable, value))

    # Display time and node list
    script.add_command('date')
    nodes_file = os.path.join(experiment_dir, 'nodes.txt')
    script.add_parallel_command('hostname > {0}'.format(nodes_file),
                                procs_per_node=1)
    script.add_command('sort --unique --output={0} {0}'.format(nodes_file))

    # Run LBANN
    for cmd in make_iterable(command):
        script.add_parallel_command(cmd)

    # Write, run, or submit batch script
    # Note: Return exit status
    if setup_only:
        script.write()
        return 0
    elif has_allocation:
        return script.run()
    else:
        return script.submit()

class LSFBatchScript(BatchScript):

    def __init__(self,
                 script_file=None,
                 work_dir=os.getcwd(),
                 nodes=1,
                 procs_per_node=1,
                 time_limit=None,
                 job_name=None,
                 partition=None,
                 account=None,
                 reservation=None,
                 launcher_args=None,
                 interpreter='/bin/bash'):
        super().__init__(script_file=script_file,
                         work_dir=work_dir,
                         interpreter=interpreter)
        self.nodes = nodes
        self.procs_per_node = procs_per_node
        self.launcher_args = launcher_args

        # Configure header with LSF job options
        self._construct_header(job_name=job_name,
                               nodes=self.nodes,
                               time_limit=time_limit,
                               partition=partition,
                               account=account,
                               reservation=reservation)

    def _construct_header(self,
                          job_name=None,
                          nodes=1,
                          time_limit=None,
                          partition=None,
                          account=None,
                          reservation=None):
        if job_name:
            self.add_header_line('#BSUB -J {}'.format(job_name))
        if partition:
            self.add_header_line('#BSUB -q {}'.format(partition))
        self.add_header_line('#BSUB --nnodes {}'.format(nodes))
        if time_limit:
            hours, minutes = divmod(int(time_limit), 60)
            self.add_header_line('#BSUB -W {}:{:02d}'.format(hours, minutes))
        self.add_header_line('#BSUB -cwd {}'.format(self.work_dir))
        self.add_header_line('#BSUB -o {}'.format(self.out_log_file))
        self.add_header_line('#BSUB -e {}'.format(self.err_log_file))
        if account:
            self.add_header_line('#BSUB -G {}'.format(account))
        if reservation:
            self.add_header_line('#BSUB -U {}'.format(reservation))

    def add_parallel_command(self,
                             command,
                             launcher_args=None,
                             nodes=None,
                             procs_per_node=None):
        if launcher_args is None:
            launcher_args = self.launcher_args
        if nodes is None:
            nodes = self.nodes
        if procs_per_node is None:
            procs_per_node = self.procs_per_node
        self.add_body_line('jsrun {0} -n {1} -r {2} {3}'
                           .format(launcher_args,
                                   nodes,
                                   procs_per_node,
                                   command))

    def submit(self):

        # Construct script file
        self.write()

        # Submit batch script and pipe output to log files
        run_proc = subprocess.Popen(['bsub', self.script_file],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    cwd=self.work_dir)
        out_proc = subprocess.Popen(['tee', self.out_log_file],
                                    stdin=run_proc.stdout,
                                    cwd=self.work_dir)
        err_proc = subprocess.Popen(['tee', self.err_log_file],
                                    stdin=run_proc.stderr,
                                    cwd=self.work_dir)
        run_proc.stdout.close()
        run_proc.stderr.close()
        run_proc.wait()
        out_proc.wait()
        err_proc.wait()
        return run_proc.returncode

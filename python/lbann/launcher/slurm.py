"""Utility functions for Slurm."""

import os, os.path
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

    Returns:
        int: Exit status from Slurm. This is really only meaningful if
            the script is run on an existing node allocation. If a
            batch job is submitted, Slurm will probably return 0
            trivially.

    """

    # Check for an existing job allocation from Slurm
    # Note: Settings for current job allocation take precedence
    has_allocation = 'SLURM_JOB_ID' in os.environ
    if has_allocation:
        job_name = os.environ['SLURM_JOB_NAME']
        partition = os.environ['SLURM_JOB_PARTITION']
        account = os.environ['SLURM_JOB_ACCOUNT']
        time_limit = None

    # Initialize Slurm batch script
    script_file = os.path.join(experiment_dir, 'batch.sh')
    script = SlurmBatchScript(script_file=script_file,
                              work_dir=experiment_dir,
                              nodes=nodes,
                              procs_per_node=procs_per_node,
                              time_limit=time_limit,
                              job_name=job_name,
                              partition=partition,
                              account=account,
                              launcher_args=srun_args)

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

class SlurmBatchScript(BatchScript):

    def __init__(self,
                 script_file=None,
                 work_dir=os.getcwd(),
                 nodes=1,
                 procs_per_node=1,
                 time_limit=None,
                 job_name=None,
                 partition=None,
                 account=None,
                 launcher_args=None,
                 interpreter='/bin/bash'):
        super().__init__(script_file=script_file,
                         work_dir=work_dir,
                         interpreter=interpreter)
        self.nodes = nodes
        self.procs_per_node = procs_per_node
        self.launcher_args = launcher_args

        # Configure header with Slurm job options
        self._construct_header(job_name=job_name,
                               nodes=self.nodes,
                               time_limit=time_limit,
                               partition=partition,
                               account=account)

    def _construct_header(self,
                          job_name=None,
                          nodes=1,
                          time_limit=None,
                          partition=None,
                          account=None):
        if job_name:
            self.add_header_line('#SBATCH --job-name={}'.format(job_name))
        self.add_header_line('#SBATCH --nodes={}'.format(nodes))
        if time_limit is not None:
            time_limit = max(time_limit, 0)
            seconds = int((time_limit % 1) * 60)
            hours, minutes = divmod(int(time_limit), 60)
            days, hours = divmod(hours, 24)
            self.add_header_line('#SBATCH --time={}-{:02d}:{:02d}:{:02d}'
                                 .format(days, hours, minutes, seconds))
        self.add_header_line('#SBATCH --workdir={}'.format(self.work_dir))
        self.add_header_line('#SBATCH --output={}'.format(self.out_log_file))
        self.add_header_line('#SBATCH --error={}'.format(self.err_log_file))
        if partition:
            self.add_header_line('#SBATCH --partition={}'.format(partition))
        if account:
            self.add_header_line('#SBATCH --account={}'.format(account))

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
        self.add_body_line('srun {0} --nodes={1} --ntasks={2} {3}'
                           .format(launcher_args,
                                   nodes,
                                   nodes * procs_per_node,
                                   command))

    def submit(self):

        # Construct script file
        self.write()

        # Submit batch script and pipe output to log files
        run_proc = subprocess.Popen(['sbatch', self.script_file],
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

"""Utility functions for Slurm."""

import os
import subprocess
from lbann.util import make_iterable
from .batch_script import BatchScript

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
                 launcher_args=[],
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
                           .format(' '.join(make_iterable(launcher_args)),
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

"""Utility functions for Slurm."""

import os
import subprocess
from lbann.util import make_iterable
from .batch_script import BatchScript

class SlurmBatchScript(BatchScript):
    """Utility class to write Slurm batch scripts."""

    def __init__(self,
                 script_file=None,
                 work_dir=os.getcwd(),
                 nodes=1,
                 procs_per_node=1,
                 time_limit=None,
                 job_name=None,
                 partition=None,
                 account=None,
                 launcher='srun',
                 launcher_args=[],
                 interpreter='/bin/bash'):
        """Construct Slurm batch script manager.

        Args:
            script_file (str): Script file.
            work_dir (str, optional): Working directory
                (default: current working directory).
            nodes (int, optional): Number of compute nodes
                (default: 1).
            procs_per_node (int, optional): Parallel processes per
                compute node (default: 1).
            time_limit (int, optional): Job time limit, in minutes
                (default: none).
            job_name (str, optional): Job name (default: none).
            partition (str, optional): Scheduler partition
                (default: none).
            account (str, optional): Scheduler account
                (default: none).
            launcher (str, optional): Parallel command launcher
                (default: srun).
            launcher_args (`Iterable` of `str`, optional):
                Command-line arguments to srun.
            interpreter (str, optional): Script interpreter
                (default: /bin/bash).

        """
        super().__init__(script_file=script_file,
                         work_dir=work_dir,
                         interpreter=interpreter)
        self.nodes = nodes
        self.procs_per_node = procs_per_node
        self.launcher = launcher
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
        """Construct script header with options for sbatch."""
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
        self.add_header_line('#SBATCH --chdir={}'.format(self.work_dir))
        self.add_header_line('#SBATCH --output={}'.format(self.out_log_file))
        self.add_header_line('#SBATCH --error={}'.format(self.err_log_file))
        if partition:
            self.add_header_line('#SBATCH --partition={}'.format(partition))
        if account:
            self.add_header_line('#SBATCH --account={}'.format(account))

    def add_parallel_command(self,
                             command,
                             launcher=None,
                             launcher_args=None,
                             nodes=None,
                             procs_per_node=None):
        """Add command to be executed in parallel.

        The command is launched with srun. Parallel processes are
        distributed evenly amongst the compute nodes.

        Args:
            command (`str` or `Iterable` of `str`s): Command to be
                executed in parallel.
            launcher (str, optional): srun executable.
            launcher_args (`Iterable` of `str`s, optional):
                Command-line arguments to srun.
            nodes (int, optional): Number of compute nodes.
            procs_per_node (int, optional): Number of parallel
                processes per compute node.

        """
        if launcher is None:
            launcher = self.launcher
        if launcher_args is None:
            launcher_args = self.launcher_args
        if nodes is None:
            nodes = self.nodes
        if procs_per_node is None:
            procs_per_node = self.procs_per_node
        args = [launcher]
        args.extend(make_iterable(launcher_args))
        args.append('--nodes={}'.format(nodes))
        args.append('--ntasks={}'.format(nodes * procs_per_node))
        args.extend(make_iterable(command))
        self.add_command(args)

    def submit(self, overwrite=False):
        """Submit batch job to Slurm with sbatch.

        The script file is written before being submitted.

        Args:
            overwrite (bool): Whether to overwrite script file if it
                already exists (default: false).

        Returns:
            int: Exit status from sbatch.

        """

        # Construct script file
        self.write(overwrite=overwrite)

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

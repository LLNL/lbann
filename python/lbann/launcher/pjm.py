"""Utility functions for PJM (Parallel Job Manager - RIKEN)."""

import os
import subprocess
import warnings
from lbann.util import make_iterable
from .batch_script import BatchScript

def _time_string(minutes):
    """Time D-hh:mm:ss format."""
    minutes = max(minutes, 0)
    seconds = int(round((minutes % 1) * 60))
    hours, minutes = divmod(int(minutes), 60)
    return f'{hours:02}:{minutes:02}:{seconds:02}'

class PJMBatchScript(BatchScript):
    """Utility class to write PJM scripts."""

    def __init__(self,
                 script_file=None,
                 work_dir=os.getcwd(),
                 nodes=1,
                 procs_per_node=1,
                 time_limit=None,
                 job_name=None,
                 partition=None,
                 launcher=None,
                 launcher_args=[],
                 interpreter='/bin/bash'):
        """Construct PJM script manager.

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
            launcher (str, optional): Parallel command launcher
                (default: mpiexec).
            launcher_args (`Iterable` of `str`, optional):
                Command-line arguments to mpiexec.
            interpreter (str, optional): Script interpreter
                (default: /bin/bash).

        """
        super().__init__(script_file=script_file,
                         work_dir=work_dir,
                         interpreter=interpreter)
        self.nodes = nodes
        self.procs_per_node = procs_per_node
        self.time_limit = time_limit
        self.job_name = job_name
        self.partition = partition
        self.launcher = launcher if launcher is not None else 'mpiexec'
        self.launcher_args = launcher_args

        # Configure header with PJM job options
        # Number of node
        self.add_header_line(f'#PJM -L "node={self.nodes}"')

        if self.time_limit is not None:
            # Job run time limit value
            self.add_header_line(f'#PJM -L "elapse={_time_string(self.time_limit)}"')
        if self.job_name is not None:
            self.add_header_line(f'#PJM --name {self.job_name}')
        if self.partition is not None:
            # Specify resource group
            self.add_header_line(f'#PJM -L "rscgrp={self.partition}"')

        self.add_header_line(f'#PJM -L "rscunit=rscunit_ft01" # Specify resource unit')
        self.add_header_line(f'#PJM --mpi "shape={self.nodes}"')
        self.add_header_line(f'#PJM --mpi "max-proc-per-node={self.procs_per_node}"')
        self.add_header_line(f'#PJM --sparam "wait-time=600"')

    def add_parallel_command(self,
                             command,
                             work_dir=None,
                             nodes=None,
                             procs_per_node=None,
                             launcher=None,
                             launcher_args=None):
        """Add command to be executed in parallel.

        The command is launched with mpiexec. Parallel processes are
        distributed evenly amongst the compute nodes.

        Args:
            command (`str` or `Iterable` of `str`s): Command to be
                executed in parallel.
            work_dir (str, optional): Working directory.
            nodes (int, optional): Number of compute nodes.
            procs_per_node (int, optional): Number of parallel
                processes per compute node.
            launcher (str, optional): mpiexec executable.
            launcher_args (`Iterable` of `str`s, optional):
                Command-line arguments to mpiexec.

        """

        # Use default values if needed
        if work_dir is None:
            work_dir = self.work_dir
        if nodes is None:
            nodes = self.nodes
        if procs_per_node is None:
            procs_per_node = self.procs_per_node
        if launcher is None:
            launcher = self.launcher
        if launcher_args is None:
            launcher_args = self.launcher_args

        # Construct mpiexec invocation
        args = [f'{launcher}']
        args.extend(make_iterable(launcher_args))
        args.extend([
            f'-n {nodes*procs_per_node}',
        ])
        args.extend([
            '-stdout-proc ./output.%j/%/1000r/stdout',
            '-stderr-proc ./output.%j/%/1000r/stderr',
        ])
        args.extend(make_iterable(command))
        self.add_command(args)

    def submit(self, overwrite=False):
        """Submit batch job.

        The script file is written before being submitted.

        Args:
            overwrite (bool): Whether to overwrite script file if it
                already exists (default: False).

        Returns:
            int: Exit status from script.

        """

        # Construct script file
        self.write(overwrite=overwrite)

        # Submit batch script and pipe output to log files
        run_proc = subprocess.Popen(['pjsub', self.script_file],
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

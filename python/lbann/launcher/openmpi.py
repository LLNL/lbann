"""Utility functions for MPI."""

import os
import warnings
from lbann.util import make_iterable
from .batch_script import BatchScript

class OpenMPIBatchScript(BatchScript):
    """Utility class to write OpenMPI scripts."""

    def __init__(self,
                 script_file=None,
                 work_dir=os.getcwd(),
                 nodes=1,
                 procs_per_node=1,
                 launcher=None,
                 launcher_args=[],
                 interpreter='/bin/bash'):
        """Construct OpenMPI script manager.

        Args:
            script_file (str): Script file.
            work_dir (str, optional): Working directory
                (default: current working directory).
            nodes (int, optional): Number of compute nodes
                (default: 1).
            procs_per_node (int, optional): Parallel processes per
                compute node (default: 1).
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
        self.launcher = launcher if launcher is not None else 'mpiexec'
        self.launcher_args = launcher_args

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
        args = [launcher]
        args.extend(make_iterable(launcher_args))
        args.extend([
            f'-n {nodes*procs_per_node}',
            f'--map-by ppr:{procs_per_node}:node',
            f'-wdir {work_dir}'
        ])
        args.extend(make_iterable(command))
        self.add_command(args)

    def submit(self, overwrite=False):
        """Submit batch job.

        OpenMPI doesn't have a notion of batch jobs, so the script is
        just run directly.

        Args:
            overwrite (bool): Whether to overwrite script file if it
                already exists (default: False).

        Returns:
            int: Exit status from script.

        """
        return self.run(overwrite)

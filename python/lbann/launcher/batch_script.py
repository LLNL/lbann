import os
import os.path
import subprocess
from lbann.util import make_iterable

class BatchScript:
    """Utility class to write batch job scripts.

    This class manages a non-interactive script file that can be
    submitted as a batch job to an HPC job scheduler. A script is made
    up of two parts: the header configures the job and the body
    contains the actual commands to be executed.

    This particular class is not fully implemented. Derived classes
    for specific job schedulers should implement
    `add_parallel_command` and `submit`, maintaining the same API.

    """

    def __init__(self,
                 script_file=None,
                 work_dir=os.getcwd(),
                 interpreter='/bin/bash'):
        """Construct batch script manager.

        Args:
            script_file (str): Script file.
            work_dir (str, optional): Working directory
                (default: current working directory).
            interpreter (str, optional): Script interpreter
                (default: /bin/bash).

        """

        # Lines in script are stored as lists of strings
        self.header = []
        self.body = []

        # Construct file paths
        self.work_dir = os.path.realpath(work_dir)
        self.script_file = script_file
        if not self.script_file:
            self.script_file = os.path.join(self.work_dir, 'batch.sh')
        self.script_file = os.path.realpath(self.script_file)
        self.out_log_file = os.path.join(self.work_dir, 'out.log')
        self.err_log_file = os.path.join(self.work_dir, 'err.log')

        # Shebang line
        if interpreter:
            self.add_header_line('#!{}'.format(interpreter))

    def add_header_line(self, line):
        """Add line to script header.

        The header should specify configuration options for the job
        scheduler, without containing executable commands.

        """
        self.header.append(line)

    def add_body_line(self, line):
        """Add line to script body.

        The body should contain the script's executable commands.

        """
        self.body.append(line)

    def add_command(self, command):
        """Add executable command to script.

        Args:
            command (`str` or `Iterable` of `str`s): Program
                invocation or sequence of program arguments.

        """
        self.add_body_line(' '.join(make_iterable(command)))

    def add_parallel_command(self,
                             command,
                             launcher=None,
                             launcher_args=None,
                             nodes=None,
                             procs_per_node=None):
        """Add command to be executed in parallel.

        The command is executed via a launcher, e.g. `mpirun`.
        Parallel processes are distributed evenly amongst the compute
        nodes.

        Args:
            command (`str` or `Iterable` of `str`s): Command to be
                executed in parallel.
            launcher (str, optional): Parallel command launcher,
               `mpirun`.
            launcher_args (`Iterable` of `str`s, optional):
                Command-line arguments to parallel command launcher.
            nodes (int, optional): Number of compute nodes.
            procs_per_node (int, optional): Number of parallel
                processes per compute node.

        """
        raise NotImplementedError(
            'classes that inherit from `BatchScript` should implement '
            '`add_parallel_command` to use a specific job scheduler'
        )

    def write(self, overwrite=False):
        """Write script to file.

        The working directory is created if needed.

        Args:
            overwrite (bool): Whether to overwrite script file if it
                already exists (default: false).

        """

        # Create directories if needed
        os.makedirs(self.work_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.script_file), exist_ok=True)

        # Check if script file already exists
        if not overwrite and os.path.isfile(self.script_file):
            raise RuntimeError('Attempted to write batch script to {}, '
                               'but it already exists'
                               .format(self.script_file))

        # Write script to file
        with open(self.script_file, 'w') as f:
            for line in self.header:
                f.write('{}\n'.format(line))
            f.write('\n')
            for line in self.body:
                f.write('{}\n'.format(line))

        # Make script file executable
        os.chmod(self.script_file, 0o755)

    def run(self, overwrite=False):
        """Execute the script.

        The script is executed directly and is _not_ submitted to a
        job scheduler. The script file is written before being
        executed.

        Args:
            overwrite (bool): Whether to overwrite script file if it
                already exists (default: false).

        Returns:
            int: Exit status from executing script.

        """

        # Construct script file
        self.write(overwrite=overwrite)

        # Run script and pipe output to log files
        run_proc = subprocess.Popen(self.script_file,
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

    def submit(self, overwrite=False):
        """Submit batch job to job scheduler.

        The script file is written before being submitted.

        Args:
            overwrite (bool): Whether to overwrite script file if it
                already exists (default: false).

        Returns:
            int: Exit status from submitting to job scheduler.

        """
        raise NotImplementedError(
            'classes that inherit from `BatchScript` should implement '
            '`submit` to use a specific job scheduler'
        )

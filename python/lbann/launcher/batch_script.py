import os
import os.path
import subprocess

class BatchScript:

    def __init__(self,
                 script_file=None,
                 work_dir=os.getcwd(),
                 interpreter='/bin/bash'):

        # Lines in script are stored as lists of strings
        # Note: The header configures batch job settings and the body
        # actually executes commands.
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
            self.header.append('#!{}'.format(interpreter))

    def add_header_line(self, line):
        self.header.append(line)

    def add_body_line(self, line):
        self.body.append(line)

    def add_command(self, command):
        self.add_body_line(command)

    def add_parallel_command(self,
                             command,
                             launcher=None,
                             launcher_args=None,
                             nodes=None,
                             procs_per_node=None):
        raise NotImplementedError()

    def write(self):

        # Create directories if needed
        os.makedirs(self.work_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.script_file), exist_ok=True)

        # Write script to file
        with open(self.script_file, 'w') as f:
            for line in self.header:
                f.write('{}\n'.format(line))
            f.write('\n')
            for line in self.body:
                f.write('{}\n'.format(line))

        # Make script file executable
        os.chmod(self.script_file, 0o755)

    def run(self):

        # Construct script file
        self.write()

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

    def submit(self):
        raise NotImplementedError()

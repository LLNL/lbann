import os, os.path
import socket
import lbann
import lbann.launcher
from lbann.util import make_iterable, nvprof_command

# ==============================================
# Detect the current compute center
# ==============================================


def is_lc_center():
    """Current system is operated by Livermore Computing at Lawrence
    Livermore National Laboratory.

    Checks whether the domain name ends with ".llnl.gov".

    """
    domain = socket.getfqdn().split('.')
    return (len(domain) > 2 and domain[-2] == 'llnl' and domain[-1] == 'gov')


def is_nersc_center():
    """Current system is operated by the National Energy Research
    Scientific Computing Center at Lawrence Berkeley National
    Laboratory.

    Checks whether the environment variable NERSC_HOST is set.

    """
    return bool(os.getenv('NERSC_HOST'))


def is_olcf_center():
    """Current system is operated by the Oak Ridge Leadership
    Computing Facility at Oak Ridge National Laboratory.

    Checks whether the domain name ends with ".ornl.gov".
    Checks whether the environment variable OLCF_MODULEPATH_ROOT is set.

    """
    domain = socket.getfqdn().split('.')
    return (len(domain) > 2 and domain[-2] == 'ornl' and domain[-1] == 'gov')


#    return bool(os.getenv('OLCF_MODULEPATH_ROOT'))


def is_riken_center():
    """Current system is operated by RIKEN.

    Checks if the system is using a Fujitsu compiler

    """
    return bool(os.getenv('FJSVXTCLANGA'))


# Detect compute center and choose launcher
_center = 'unknown'
launcher = lbann.launcher
if is_lc_center():
    _center = 'lc'
    import lbann.contrib.lc.systems
    if lbann.contrib.lc.systems.is_lc_system():
        import lbann.contrib.lc.launcher
        launcher = lbann.contrib.lc.launcher
elif is_nersc_center():
    _center = 'nersc'
    import lbann.contrib.nersc.systems
    if lbann.contrib.nersc.systems.is_nersc_system():
        import lbann.contrib.nersc.launcher
        launcher = lbann.contrib.nersc.launcher
elif is_olcf_center():
    _center = 'olcf'
    import lbann.contrib.olcf.systems
    if lbann.contrib.olcf.systems.is_olcf_system():
        import lbann.contrib.olcf.launcher
        launcher = lbann.contrib.olcf.launcher
elif is_riken_center():
    _center = 'riken'
    import lbann.contrib.riken.systems
    if lbann.contrib.riken.systems.is_riken_system():
        import lbann.contrib.riken.launcher
        launcher = lbann.contrib.riken.launcher


def compute_center():
    """Name of organization that operates current system."""
    return _center


# ==============================================
# Launcher functions
# ==============================================


def run(
    trainer,
    model,
    data_reader,
    optimizer,
    lbann_exe=lbann.lbann_exe(),
    lbann_args=[],
    procs_per_trainer=None,
    overwrite_script=False,
    setup_only=False,
    batch_job=False,
    proto_file_name=None,
    nvprof=False,
    nvprof_output_name=None,
    binary_protobuf=False,
    *args,
    **kwargs,
):
    """Run LBANN with system-specific optimizations.

    This is intended to match the behavior of `lbann.run`, with
    defaults and optimizations for the current system. See that
    function for a full list of options.

    """

    # Create batch script generator
    script = make_batch_script(*args, **kwargs)

    # Batch script prints start time
    script.add_command('echo "Started at $(date)"')

    # Set default file name and extension
    if proto_file_name is None:
        proto_file_name = ('experiment.protobin'
                           if binary_protobuf else 'experiment.prototext')

    # Batch script invokes LBANN
    lbann_command = [lbann_exe]
    if nvprof:
        lbann_command = nvprof_command(
            work_dir=script.work_dir,
            output_name=nvprof_output_name) + lbann_command
    lbann_command.extend(make_iterable(lbann_args))
    proto_file = os.path.join(script.work_dir, proto_file_name)
    lbann.proto.save_prototext(proto_file,
                               binary=binary_protobuf,
                               trainer=trainer,
                               model=model,
                               data_reader=data_reader,
                               optimizer=optimizer)
    lbann_command.append('--prototext={}'.format(proto_file))
    if procs_per_trainer is not None:
        lbann_command.append(f'--procs_per_trainer={procs_per_trainer}')
    script.add_parallel_command(lbann_command)
    script.add_command('status=$?')

    # Batch script prints finish time and returns status
    script.add_command('echo "Finished at $(date)"')
    script.add_command('exit ${status}')

    # Write, run, or submit batch script
    status = 0
    if setup_only:
        script.write(overwrite=overwrite_script)
    elif batch_job:
        status = script.submit(overwrite=overwrite_script)
    else:
        status = script.run(overwrite=overwrite_script)
    return status


def make_batch_script(*args, **kwargs):
    """Construct batch script manager with system-specific optimizations.

    This is intended to match the behavior of
    `lbann.launcher.make_batch_script`, with defaults and
    optimizations for the current system.

    """
    return launcher.make_batch_script(*args, **kwargs)

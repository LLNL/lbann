import os, os.path
import socket
import lbann
import lbann.launcher
from lbann.util import make_iterable


# ==============================================
# Identify the Compute Center
# ==============================================

# Supported Compute Centers
NERSC = 'NERSC'
LC = 'LC'

def is_nersc_center():
    NERSC_HOST = os.getenv('NERSC_HOST')
    if NERSC_HOST:
        return NERSC
    else:
        return None

def is_lc_center():
    _dn = str.join('.', socket.getfqdn().split('.')[-2:])
    if _dn == 'llnl.gov':
        return LC
    else:
        return None

# Detect center
_center = is_lc_center()
if not _center:
    _center = is_nersc_center()

if not _center:
    system = socket.getfqdn()
    raise RuntimeError('unknown compute center (' + system + ')')

if _center == LC:
    import lbann.contrib.lc.launcher
    import lbann.contrib.lc.systems
elif _center == NERSC:
    import lbann.contrib.nersc.launcher
    import lbann.contrib.nersc.systems


def run(
    trainer,
    model,
    data_reader,
    optimizer,
    lbann_exe=lbann.lbann_exe(),
    lbann_args=[],
    overwrite_script=False,
    setup_only=False,
    batch_job=False,
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

    # Batch script invokes LBANN
    lbann_command = [lbann_exe]
    lbann_command.extend(make_iterable(lbann_args))
    prototext_file = os.path.join(script.work_dir, 'experiment.prototext')
    lbann.proto.save_prototext(prototext_file,
                               trainer=trainer,
                               model=model,
                               data_reader=data_reader,
                               optimizer=optimizer)
    lbann_command.append('--prototext={}'.format(prototext_file))
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

    print('Starting to make the batch script')
    # Livermore Computing
    if _center == LC and lbann.contrib.lc.systems.is_lc_system():
        return lbann.contrib.lc.launcher.make_batch_script(*args, **kwargs)

    # NERSC
    elif _center == NERSC and lbann.contrib.nersc.systems.is_nersc_system():
        return lbann.contrib.nersc.launcher.make_batch_script(*args, **kwargs)

    # Default launcher
    return lbann.launcher.make_batch_script(*args, **kwargs)

"""Default settings for RIKEN systems."""
import socket
import re
import platform
import os

# ==============================================
# Set system parameters
# ==============================================

class SystemParams:
    """Simple data structure to describe an LC system."""
    def __init__(self, cores_per_node, gpus_per_node, scheduler):
        self.cores_per_node = cores_per_node
        self.gpus_per_node = gpus_per_node
        self.scheduler = scheduler

# Supported LC systems
_system_params = {
    'fugaku':   SystemParams(48, 0, 'pjm'),
}

# Detect system
#_system = re.sub(r'\d+', '', socket.gethostname())

# ==============================================
# Access functions
# ==============================================

def system():
    """Name of system.

    Hostname with trailing digits removed.

    """
    if re.match(r'aarch64', platform.machine()):
        return 'fugaku'
    elif bool(os.getenv('FJSVXTCLANGA')):
        return 'fugaku'

    return 'unknown'

def is_riken_system(system = system()):
    """Whether current system is a supported RIKEN system."""
    return system in _system_params.keys()

def gpus_per_node(system = system()):
    """Number of GPUs per node."""
    if not is_riken_system(system):
        raise RuntimeError('unknown system (' + system + ')')
    return _system_params[system].gpus_per_node

def has_gpu(system = system()):
    """Whether RIKEN system has GPUs."""
    return gpus_per_node(system) > 0

def cores_per_node(system = system()):
    """Number of CPU cores per node."""
    if not is_riken_system(system):
        raise RuntimeError('unknown system (' + system + ')')
    return _system_params[system].cores_per_node

def scheduler(system = system()):
    """Job scheduler for RIKEN system."""
    if not is_riken_system(system):
        raise RuntimeError('unknown system (' + system + ')')
    return _system_params[system].scheduler

def procs_per_node(system = system()):
    """Default number of processes per node."""
    if has_gpu(system):
        return gpus_per_node(system)
    else:
        # Fugaku and FX700 have 4 NUMA domains
        return 4

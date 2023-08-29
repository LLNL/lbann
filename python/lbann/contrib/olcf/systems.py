"""Default settings for OLCF systems."""
import socket
import re

# ==============================================
# Set system parameters
# ==============================================

class SystemParams:
    """Simple data structure to describe an OLCF system."""
    def __init__(self, cores_per_node, gpus_per_node, scheduler):
        self.cores_per_node = cores_per_node
        self.gpus_per_node = gpus_per_node
        self.scheduler = scheduler

# Supported LC systems
_system_params = {
    'summit':   SystemParams(44, 6, 'lsf'),
    'frontier': SystemParams(64, 8, 'slurm'),
    'crusher': SystemParams(64, 8, 'slurm'),
}

# Detect system
_system = re.sub(r'\d+', '', socket.gethostname())

# ==============================================
# Access functions
# ==============================================

def system():
    """Name of system.

    Hostname with trailing digits removed.

    """
    return _system

def is_olcf_system(system = system()):
    """Whether current system is a supported OLCF system."""
    return _system in _system_params.keys()

def gpus_per_node(system = system()):
    """Number of GPUs per node."""
    if not is_olcf_system(system):
        raise RuntimeError('unknown system (' + system + ')')
    return _system_params[system].gpus_per_node

def has_gpu(system = system()):
    """Whether OLCF system has GPUs."""
    return gpus_per_node(system) > 0

def cores_per_node(system = system()):
    """Number of CPU cores per node."""
    if not is_olcf_system(system):
        raise RuntimeError('unknown system (' + system + ')')
    return _system_params[system].cores_per_node

def scheduler(system = system()):
    """Job scheduler for OLCF system."""
    if not is_olcf_system(system):
        raise RuntimeError('unknown system (' + system + ')')
    return _system_params[system].scheduler

def procs_per_node(system = system()):
    """Default number of processes per node."""
    if has_gpu(system):
        return gpus_per_node(system)
    else:
        raise RuntimeError('unknown system (' + system + ')')

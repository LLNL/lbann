"""Default settings for LC systems."""
import socket
import re

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
    'catalyst': SystemParams(24, 0, 'slurm'),
    'corona':   SystemParams(48, 8, 'flux'),
    'lassen':   SystemParams(44, 4, 'lsf'),
    'pascal':   SystemParams(36, 2, 'slurm'),
    'quartz':   SystemParams(36, 0, 'slurm'),
    'rzansel':  SystemParams(44, 4, 'lsf'),
    'rzvernal': SystemParams(64, 8, 'flux'),
    'sierra':   SystemParams(44, 4, 'lsf'),
    'tioga':    SystemParams(64, 8, 'flux'),
    'tuolumne': SystemParams(96, 4, 'flux'),
    'rzadams':  SystemParams(96, 4, 'flux'),
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

def is_lc_system(system = system()):
    """Whether current system is a supported LC system."""
    return _system in _system_params.keys()

def gpus_per_node(system = system()):
    """Number of GPUs per node."""
    if not is_lc_system(system):
        raise RuntimeError('unknown system (' + system + ')')
    return _system_params[system].gpus_per_node

def has_gpu(system = system()):
    """Whether LC system has GPUs."""
    return gpus_per_node(system) > 0

def cores_per_node(system = system()):
    """Number of CPU cores per node."""
    if not is_lc_system(system):
        raise RuntimeError('unknown system (' + system + ')')
    return _system_params[system].cores_per_node

def scheduler(system = system()):
    """Job scheduler for LC system."""
    if not is_lc_system(system):
        raise RuntimeError('unknown system (' + system + ')')
    return _system_params[system].scheduler

def procs_per_node(system = system()):
    """Default number of processes per node."""
    if has_gpu(system):
        return gpus_per_node(system)
    else:
        # Catalyst and Quartz have 2 sockets per node
        ### @todo Think of a smarter heuristic
        return 2

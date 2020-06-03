"""Default settings for LC systems."""
import socket
import re

# ==============================================
# Set system parameters
# ==============================================

class SystemParams:
    """Simple data structure to describe an NERSC system."""
    def __init__(self, cores_per_node, numa_nodes_per_node, gpus_per_node, scheduler):
        self.cores_per_node = cores_per_node
        self.numa_nodes_per_node = numa_nodes_per_node
        self.gpus_per_node = gpus_per_node
        self.scheduler = scheduler

# Supported LC systems
CORI_GPU='cgpu'
_system_params = {
    CORI_GPU : SystemParams(40, 2, 8, 'slurm'),
}

# Detect system
_system = re.sub(r'\d+', '', socket.gethostname())
if _system not in _system_params.keys():
    _system = None

# ==============================================
# Access functions
# ==============================================

def system():
    """Name of NERSC system."""
    if _system:
        return _system
    else:
        raise RuntimeError('unknown system '
                           '(' + socket.gethostname() + ')')

def is_nersc_system(system = system()):
    """Whether current system is a supported NERSC system."""
    return (system is not None) and (system in _system_params.keys())

def gpus_per_node(system = system()):
    """Number of GPUs per node."""
    if not is_nersc_system(system):
        raise RuntimeError('unknown system (' + system + ')')
    return _system_params[system].gpus_per_node

def has_gpu(system = system()):
    """Whether NERSC system has GPUs."""
    return gpus_per_node(system) > 0

def cores_per_node(system = system()):
    """Number of CPU cores per node."""
    if not is_nersc_system(system):
        raise RuntimeError('unknown system (' + system + ')')
    return _system_params[system].cores_per_node

def numa_nodes_per_node(system = system()):
    """Number of NUMA nodes per node."""
    if not is_nersc_system(system):
        raise RuntimeError('unknown system (' + system + ')')
    return _system_params[system].numa_nodes_per_node

def scheduler(system = system()):
    """Job scheduler for NERSC system."""
    if not is_nersc_system(system):
        raise RuntimeError('unknown system (' + system + ')')
    return _system_params[system].scheduler

def procs_per_node(system = system()):
    """Default number of processes per node."""
    if has_gpu(system):
        return gpus_per_node(system)
    else:
        ### @todo Think of a smarter heuristic
        return numa_nodes_per_node(system)

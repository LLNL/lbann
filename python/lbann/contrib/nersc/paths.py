"""Useful file paths on NERSC systems."""
import os.path
from lbann.contrib.nersc.systems import system

# ==============================================
# Data sets
# ==============================================

def parallel_file_system_path(system = system()):
    """Base path to parallel file system."""
    if system == 'cgpu':
        return '/global/cfs/cdirs/m3363/'
    else:
        raise RuntimeError('unknown parallel file system path on ' + system)

def imagenet_dir(system = system(), data_set = 'training'):
    """ImageNet directory on NERSC system.

    The directory contains JPEG images from the ILSVRC2012
    competition. File names in the label file are relative to this
    directory. The images can be obtained from
    http://image-net.org/challenges/LSVRC/2012/.

    There are three available data sets: 'training', 'validation', and
    'testing'.

    """
    raise RuntimeError('ImageNet data is not available on ' + system)

def imagenet_labels(system = system(), data_set = 'train'):
    """ImageNet label file on NERSC system.

    The file contains ground truth labels from the ILSVRC2012
    competition. It is a plain text file where each line contains an
    image file path (relative to the ImageNet directory; see the
    `imagenet_dir` function) and the corresponding label ID.

    There are three available data sets: 'training', 'validation', and
    'testing'.

    """
    raise RuntimeError('ImageNet data is not available on ' + system)

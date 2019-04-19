"""Useful file paths on LC systems."""
import os.path
from lbann.util import make_iterable, lbann_dir
from lbann.contrib.lc.systems import system

# ==============================================
# File paths
# ==============================================

def install_dir(build_type = None, system = system()):
    """LBANN install directory.

    Searches in the `build` directory. Assumes LBANN has been built
    with `scripts/build_lbann_lc.sh`.

    """
    if not build_type:
        build_type = ('Release', 'Debug')
    for _type in make_iterable(build_type):
        _dir = os.path.join(lbann_dir(),
                            'build',
                            'gnu.{}.{}.llnl.gov'.format(_type, system),
                            'install')
        if os.path.isdir(_dir):
            return _dir
    raise RuntimeError('could not find install directory')

def lbann_exe(build_type = None, system = system()):
    """LBANN executable."""
    return os.path.join(install_dir(build_type, system), 'bin', 'lbann')

def parallel_fs_base_path(system):
    """Base path to parallel file system."""
    if system in ('lassen', 'sierra'):
        return '/p/gpfs1/'
    else:
        return '/p/lustre2/'

# ==============================================
# Data sets
# ==============================================

def mnist_dir(system = system()):
    """MNIST directory on LC system.

    The directory contains four files: train-images-idx3-ubyte,
    train-labels-idx1-ubyte, t10k-images-idx3-ubyte,
    t10k-labels-idx1-ubyte. These files can be obtained by downloading
    from http://yann.lecun.com/exdb/mnist/ and uncompressing.

    """
    return parallel_fs_base_path(system) + 'brainusr/datasets/MNIST'

def imagenet_dir(system = system(), data_set = 'training',
                 num_classes = 1000):
    """ImageNet directory on LC system.

    The directory contains JPEG images from the ILSVRC2012
    competition. File names in the label file are relative to this
    directory. The images can be obtained from
    http://image-net.org/challenges/LSVRC/2012/.

    There are three available data sets: 'training', 'validation', and
    'testing'.

    Some of these data sets have been preprocessed to only include
    images in a subset of the label classes, e.g. images in the first
    10 label classes. This is convenient for quickly evaluating
    performance or learning behavior. The availabiilty of these
    subsampled data sets may vary by system.

    """
    base_path = parallel_fs_base_path(system)
    base_path += 'brainusr/datasets/ILSVRC2012/original/'
    if data_set.lower() in ('train', 'training'):
        return base_path + 'train/'
    elif data_set.lower() in ('val', 'validation'):
        return base_path + 'val/'
    elif data_set.lower() in ('test', 'testing'):
        return base_path + 'test/'
    else:
        raise RuntimeError('unknown ImageNet data set (' + data_set + ')')

def imagenet_labels(system = system(), data_set = 'train',
                    num_classes = 1000):
    """ImageNet label file on LC system.

    The file contains ground truth labels from the ILSVRC2012
    competition. It is a plain text file where each line contains an
    image file path (relative to the ImageNet directory; see the
    `imagenet_dir` function) and the corresponding label ID.

    There are three available data sets: 'training', 'validation', and
    'testing'.

    Some of these data sets have been preprocessed to only include
    images in a subset of the label classes, e.g. images in the first
    10 label classes. This is convenient for quickly evaluating
    performance or learning behavior. The availabiilty of these
    subsampled data sets may vary by system.

    """
    label_dir = parallel_fs_base_path(system)
    label_dir += 'brainusr/datasets/ILSVRC2012/labels/'
    suffixes = {1000: '', 10: '_c0-9', 100: '_c0-99',
                200: '_c100-299', 300: '_c0-299'}
    if data_set.lower() in ('train', 'training'):
        if num_classes in suffixes.keys():
            return os.path.join(label_dir,
                                'train' + suffixes[num_classes] + '.txt')
        else:
            raise RuntimeError('invalid number of classes ({0}) '
                               'for ImageNet data set ({1})'
                               .format(num_classes, data_set))
    elif data_set.lower() in ('val', 'validation'):
        if num_classes in suffixes.keys():
            return os.path.join(label_dir,
                                'val' + suffixes[num_classes] + '.txt')
        else:
            raise RuntimeError('invalid number of classes ({0}) '
                               'for ImageNet data set ({1})'
                               .format(num_classes, data_set))
    elif data_set.lower() in ('test', 'testing'):
        return os.path.join(label_dir, 'test.txt')
    else:
        raise RuntimeError('unknown ImageNet data set (' + data_set + ')')

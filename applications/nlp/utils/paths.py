"""Useful file paths."""
import os
import os.path
import re
import socket


def system():
    """Name of current compute system.

    Primarily used to detect LLNL LC systems.

    """
    return re.sub(r'\d+', '', socket.gethostname())


def root_dir():
    """Root directory for LBANN NLP application."""
    return os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

def wmt_dir():
    """Data directory for the WMT 2016 dataset.

    See https://huggingface.co/datasets/wmt16

    The dataset has already been downloaded on LLNL LC systems and is
    available to anyone in the "lbann" group.

    """

    return '/p/vast1/lbann/datasets/wmt16_en_de_huggingface'

def wmt_dir_old(system=system()):
    """Data directory for the WMT 2014 dataset.

    See https://pytorchnlp.readthedocs.io/en/latest/source/torchnlp.datasets.html#torchnlp.datasets.wmt_dataset.

    The dataset has already been downloaded on LLNL LC systems and is
    available to anyone in the "brainusr" group. If the dataset is not
    accessible, a path within the application directory is returned.

    """

    # Cached datasets on LC systems
    path = None
    if system in ('lassen', 'sierra'):
        path = '/p/gpfs1/brainusr/datasets/wmt16_en_de'
    elif system in ('pascal', 'catalyst', 'quartz', 'surface'):
        path = '/p/vast1/lbann/datasets/wmt16_en_de'

    # Default path if cached dataset isn't available
    if not path or not os.access(path, os.R_OK):
        path = os.path.join(root_dir(), 'data', 'wmt16_en_de')

    return path

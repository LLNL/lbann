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

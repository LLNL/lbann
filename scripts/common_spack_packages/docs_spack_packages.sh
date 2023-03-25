#!/bin/sh

# Add packages used by the LBANN documentation
spack add py-breathe ${CENTER_COMPILER}
spack add py-sphinx-rtd-theme ${CENTER_COMPILER}
spack add doxygen ${CENTER_COMPILER}
spack add py-m2r ${CENTER_COMPILER}

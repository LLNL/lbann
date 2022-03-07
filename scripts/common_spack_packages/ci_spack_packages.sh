#!/bin/sh

# Add packages used by the LBANN CI and applications
# Note that py-numpy@1.16.0: is explicitly added by the build_lbann.sh
# script when the lbann+python variant is enabled
spack add python
spack add py-pytest
spack add py-scipy
spack add py-tqdm

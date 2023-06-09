#!/bin/sh

# Add packages used by the LBANN CI and applications
# Note that py-numpy@1.16.0: is explicitly added by the build_lbann.sh
# script when the lbann+python variant is enabled
# spack add python ${CENTER_COMPILER}
# spack add py-pytest ${CENTER_COMPILER}
# spack add py-scipy@1.8.1 ${CENTER_COMPILER}
# spack add py-tqdm ${CENTER_COMPILER}
export LBANN_EXTRA_PKGS="python py-pytest py-scipy@1.8.1 py-tqdm"

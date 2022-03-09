#!/bin/sh

# Add packages used by the LBANN CI and applications
spack add py-numpy@1.16.0:
spack add py-pytest
spack add py-scipy
spack add py-tqdm

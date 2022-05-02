#!/bin/sh

# Add packages used by the ATOM project
spack add py-pandas ${CENTER_COMPILER}
spack add python@3.9.1: ${CENTER_COMPILER}
spack add py-torch@1.7.1 ${CENTER_COMPILER} ${DEPENDENT_PACKAGES_GPU_VARIANTS}
spack add py-scikit-learn ${CENTER_COMPILER}
spack add py-tqdm ${CENTER_COMPILER}
spack add py-nltk ${CENTER_COMPILER}
spack add rdkit ${CENTER_COMPILER}

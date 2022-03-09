#!/bin/sh

# Add packages used by the ATOM project
spack add py-pandas
spack add python@3.9.1:
spack add py-torch@1.7.1 ${DEPENDENT_PACKAGES_GPU_VARIANTS}
spack add py-scipy
spack add py-scikit-learn
spack add py-tqdm
spack add py-nltk
spack add rdkit

#!/bin/sh

# This script is expected to be called from build_lbann.sh or to have the following environment variables defined
# LBANN_HOME
# LBANN_BUILD_DIR
# SPACK_ENV_DIR
# CENTER

temp_file=$(mktemp)

# Defines STD_PACKAGES and STD_MODULES
source ${SPACK_ENV_DIR}/std_versions_and_variants.sh
# Defines EXTERNAL_ALL_PACKAGES and EXTERNAL_PACKAGES
source ${SPACK_ENV_DIR}/${CENTER}/externals-${SPACK_ARCH}.sh
# Defines COMPILER_ALL_PACKAGES and COMPILER_DEFINITIONS
source ${SPACK_ENV_DIR}/${CENTER}/compilers.sh

if [[ ${SYS} != "Darwin" ]]; then
COMPILER_PACKAGE=$(cat <<EOF
  - gcc
EOF
)
fi

if [[ "${ENABLE_GPUS}" == "ON" ]]; then
GPU_PACKAGES=$(cat <<EOF
  - cudnn
  - cub
  - nccl
EOF
)
fi

SPACK_ENV=$(cat <<EOF
spack:
  concretization: together
  specs:
  - aluminum
  - catch2
  - cereal
  - clara
  - cmake
  - cnpy
  - conduit
  - half
  - hwloc
  - hydrogen
${COMPILER_PACKAGE}
  - opencv
  - ninja
  - zlib
${GPU_PACKAGES}
  - py-numpy
  - py-protobuf
  - py-pytest
  - py-setuptools
  mirrors: {}
  modules:
    enable: []
  repos: []
  config: {}

  packages:
${EXTERNAL_ALL_PACKAGES}
${COMPILER_ALL_PACKAGES}

${EXTERNAL_PACKAGES}

${STD_PACKAGES}

    aluminum:
      buildable: true
      version: [master]
      variants: +gpu +nccl ~mpi_cuda
      providers: {}
      paths: {}
      modules: {}
      compiler: []
      target: []
    hydrogen:
      buildable: true
      version: [develop]
      variants: +openmp_blas +shared +int64 +al
      providers: {}
      paths: {}
      modules: {}
      compiler: []
      target: []

${COMPILER_DEFINITIONS}

${STD_MODULES}
  view: true
EOF
)

echo "${SPACK_ENV}" > ${temp_file}

if [[ $(spack env list | grep ${LBANN_ENV}) ]]; then
    echo "Spack environment ${LBANN_ENV} already exists... overwriting it"
    CMD="spack env rm ${LBANN_ENV}"
    echo ${CMD}
    ${CMD}
fi

CMD="spack env create ${LBANN_ENV} ${temp_file}"
#CMD="spack env create -d ${LBANN_BUILD_DIR} ${temp_file}"
echo ${CMD}
${CMD}
#cd ${LBANN_BUILD_DIR}
CMD="spack env activate -p ${LBANN_ENV}"
echo ${CMD}
${CMD}
echo "********************************************************************************"
echo "* WARNING a fresh installation of the spack dependencies can take a long time   "
echo "********************************************************************************"
CMD="spack install"
echo ${CMD}
${CMD}
RESULT=$?
if [ $RESULT -ne 0 ]; then
    echo "Spack installation failed"
    exit -1
fi
CMD="spack env loads" # Spack creates a file named loads that has all of the correct modules
echo ${CMD}
${CMD}
source ${SPACK_ROOT}/share/spack/setup-env.sh # Rerun setup since spack doesn't modify MODULEPATH unless there are module files defined
CMD="source ${PWD}/loads"
echo ${CMD}
${CMD}

#!/bin/bash

# "spack" is just a shell function; it may not be exported to this
# scope. Just to be sure, reload the shell integration.
if [ -n "${SPACK_ROOT}" ]; then
    source ${SPACK_ROOT}/share/spack/setup-env.sh
fi

SPACK_VERSION=$(spack --version | sed 's/-.*//g')
MIN_SPACK_VERSION=0.15.4

source $(dirname ${BASH_SOURCE})/utilities.sh

compare_versions ${SPACK_VERSION} ${MIN_SPACK_VERSION}
VALID_SPACK=$?

if [[ ${VALID_SPACK} -eq 2 ]]; then
    echo "Newer version of Spack required.  Detected version ${SPACK_VERSION} requires at least ${MIN_SPACK_VERSION}"
    exit 1
fi

# Detect system parameters
CLUSTER=$(hostname | sed 's/\([a-zA-Z][a-zA-Z]*\)[0-9]*/\1/g')
ARCH=$(uname -m)
SYS=$(uname -s)

SCRIPT=${BASH_SOURCE}

if [[ ${SYS} = "Darwin" ]]; then
SCRIPTS_DIR=$(normpath $(dirname $(osx_realpath ${SCRIPT})))
else
SCRIPTS_DIR=$(realpath $(dirname ${SCRIPT}))
fi

LBANN_HOME=$(dirname ${SCRIPTS_DIR})
SPACK_ENV_DIR=${LBANN_HOME}/spack_environments

# Identify the center that we are running at
CENTER=
if [[ ${SYS} = "Darwin" ]]; then
    CENTER="osx"
else
    CORI=$([[ $(hostname) =~ (cori|cgpu) ]] && echo 1 || echo 0)
    DOMAINNAME=$(python -c 'import socket; domain = socket.getfqdn().split("."); print(domain[-2] + "." + domain[-1])')
    if [[ ${CORI} -eq 1 ]]; then
        CENTER="nersc"
        # Make sure to purge and setup the modules properly prior to finding the Spack architecture
        source ${SPACK_ENV_DIR}/${CENTER}/setup_modules.sh
    elif [[ ${DOMAINNAME} = "ornl.gov" ]]; then
        CENTER="olcf"
    elif [[ ${DOMAINNAME} = "llnl.gov" ]]; then
        CENTER="llnl_lc"
    else
        CENTER="llnl_lc"
    fi
fi

SPACK_ARCH=$(spack arch)
SPACK_ARCH_TARGET=$(spack arch -t)

SCRIPT=$(basename ${BASH_SOURCE})
BUILD_DIR=${LBANN_HOME}/build/spack
ENABLE_GPUS=ON
GPU_VARIANTS="+cuda+nccl"
ENABLE_HALF=OFF
HALF_VARIANTS="~half"
BUILD_TYPE=Release
VERBOSE=0
LBANN_ENV=
SPACK_INSTALL_ARGS=
BUILD_LBANN_SW_STACK="TRUE"

################################################################
# Help message
################################################################

function help_message {
    local SCRIPT=$(basename ${BASH_SOURCE})
    local N=$(tput sgr0)    # Normal text
    local C=$(tput setf 4)  # Colored text
    cat << EOF
Build LBANN on an LLNL LC system.
Can be called anywhere in the LBANN project tree.
Usage: ${SCRIPT} [options]
Options:
  ${C}--help${N}               Display this help message and exit.
  ${C}--verbose${N}            Verbose output.
  ${C}-d | -deps-only)${N}     Only install the lbann dependencies
  ${C}-e | --env${N}           Build and install LBANN in the spack environment provided.
  ${C}--half${N}               Enable support for HALF precision data types in Hydrogen and DiHydrogen
  ${C}--disable-gpus${N}       Disable GPUS
  ${C}-s | --superbuild${N}    Superbuild LBANN with dihydrogen, hydrogen, and aluminum
EOF
}

################################################################
# Parse command-line arguments
################################################################

while :; do
    case ${1} in
        -h|--help)
            # Help message
            help_message
            exit 1
            ;;
        -d|-deps-only)
            DEPS_ONLY="TRUE"
# Until several spack bugs are fixed we cannot use this flag
#            SPACK_INSTALL_ARGS="--only dependencies"
            ;;
        -e|--env)
            # Change default build directory
            if [ -n "${2}" ]; then
                LBANN_ENV=${2}
                shift
            else
                echo "\"${1}\" option requires a non-empty option argument" >&2
                exit 1
            fi
            ;;
        --half)
            ENABLE_HALF=ON
            HALF_VARIANTS="+half"
            ;;
        --disable-gpus)
            ENABLE_GPUS=OFF
            GPU_VARIANTS=
            ;;
        -s|--superbuild)
            BUILD_LBANN_SW_STACK="FALSE"
            ;;
        -?*)
            # Unknown option
            echo "Unknown option (${1})" >&2
            exit 1
            ;;
        *)
            # Break loop if there are no more options
            break
    esac
    shift
done

temp_file=$(mktemp)

# Defines STD_PACKAGES and STD_MODULES
source ${SPACK_ENV_DIR}/std_versions_and_variants.sh
# Defines EXTERNAL_ALL_PACKAGES and EXTERNAL_PACKAGES
source ${SPACK_ENV_DIR}/${CENTER}/externals-${SPACK_ARCH}.sh
# Defines COMPILER_ALL_PACKAGES and COMPILER_DEFINITIONS
source ${SPACK_ENV_DIR}/${CENTER}/compilers.sh

# Disable GPU features on OS X
if [[ ${SYS} = "Darwin" ]]; then
    ENABLE_GPUS=OFF
    GPU_VARIANTS=
fi

BUILD_SPECS=
HYDROGEN_VARIANTS="variants: +shared +int64 +al ${HALF_VARIANTS}"
DIHYDROGEN_VARIANTS="variants: +shared +al +openmp ${HALF_VARIANTS}"
if [[ ${DEPS_ONLY} = "TRUE" ]]; then
    if [[ ${SYS} != "Darwin" ]]; then
        HYDROGEN_VARIANTS="${HYDROGEN_VARIANTS} +openmp_blas"
        DIHYDROGEN_VARIANTS="${DIHYDROGEN_VARIANTS} +openmp_blas"
        COMPILER_PACKAGE=$(cat <<EOF
  - gcc
EOF
)
    else
        HYDROGEN_VARIANTS="${HYDROGEN_VARIANTS} blas=accelerate"
        DIHYDROGEN_VARIANTS="${DIHYDROGEN_VARIANTS} blas=accelerate"
        COMPILER_PACKAGE=$(cat <<EOF
  - llvm
EOF
)
    fi

    GPU_PACKAGES=
    if [[ "${ENABLE_GPUS}" == "ON" ]]; then
        GPU_PACKAGES=$(cat <<EOF
  - cudnn
  - cuda
  - nccl
EOF
)
    fi

    HALF_PACKAGES=
    if [[ "${ENABLE_HALF}" == "ON" ]]; then
        HALF_PACKAGES=$(cat <<EOF
  - half
EOF
)
    fi

    SUPERBUILD_SPECS=
    if [[ ${BUILD_LBANN_SW_STACK} == "TRUE" ]]; then
        SUPERBUILD_SPECS=$(cat <<EOF
  - aluminum
  - hydrogen
  - dihydrogen
EOF
)
    fi

    # Include additional specs if only the dependencies are build
    BUILD_SPECS=$(cat <<EOF
# These packages should go away when spack fixes its environments
${SUPERBUILD_SPECS}
  - cereal
  - clara
  - cnpy
  - conduit
${HALF_PACKAGES}
  - hwloc
  - opencv
  - zlib
${GPU_PACKAGES}
  - py-numpy
  - py-protobuf
  - py-setuptools
  - mpi

# These are required
  - catch2
  - cmake
  - ninja
  - python
  - py-pytest
${COMPILER_PACKAGE}
EOF
)
    LBANN_ENV="${LBANN_ENV:-lbann-dev-${SPACK_ARCH_TARGET}}"
else
    LBANN_ENV="${LBANN_ENV:-lbann-${SPACK_ARCH_TARGET}}"
    BUILD_SPECS=$(cat <<EOF
  - lbann@develop${GPU_VARIANTS}
EOF
)
fi

AL_VARIANTS=
if [[ "${ENABLE_GPUS}" == "ON" ]]; then
#    CUDA_ARCH="cuda_arch=60,61,62,70"
    AL_VARIANTS="variants: +cuda +nccl +ht +cuda_rma"
    HYDROGEN_VARIANTS="${HYDROGEN_VARIANTS} +cuda"
    DIHYDROGEN_VARIANTS="${DIHYDROGEN_VARIANTS} +cuda +legacy"
fi

SPACK_ENV=$(cat <<EOF
spack:
  concretization: together
  specs:
${BUILD_SPECS}
  packages:
${EXTERNAL_ALL_PACKAGES}
${COMPILER_ALL_PACKAGES}
${EXTERNAL_PACKAGES}
${STD_PACKAGES}
    aluminum:
      buildable: true
      version:
      - 0.4.0
      ${AL_VARIANTS}
      providers: {}
      compiler: []
      target: []
    hydrogen:
      buildable: true
      version:
      - 1.4.0
      ${HYDROGEN_VARIANTS}
      providers: {}
      compiler: []
      target: []
    dihydrogen:
      buildable: true
      version:
      - master
      ${DIHYDROGEN_VARIANTS}
      providers: {}
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
echo ${CMD}
${CMD}

CMD="spack env activate -p ${LBANN_ENV}"
echo ${CMD}
${CMD}

CMD="spack install ${SPACK_INSTALL_ARGS}"
echo ${CMD}
eval ${CMD}
if [[ $? -ne 0 ]]; then
    echo "--------------------"
    echo "Spack installation FAILED"
    echo "--------------------"
    exit 1
else
    if [[ ${DEPS_ONLY} = "TRUE" ]]; then
        echo "LBANN's dependencies are installed in a spack environment named ${LBANN_ENV}, access it via:"
        echo "  spack env activate -p ${LBANN_ENV}"
        # Reactivate the spack environment since a clean installation will note setup the modules properly
        CMD=". $SPACK_ROOT/share/spack/setup-env.sh"
        ${CMD}
        # It is no longer necessary to load modules
        # CMD="spack env loads"
        # ${CMD}

        echo "Build LBANN from source using the spack environment ${LBANN_ENV}, using the build script:"
        echo "  ${SCRIPTS_DIR}/build_lbann_from_source.sh -e ${LBANN_ENV}"
    else
        echo "LBANN is installed in a spack environment named ${LBANN_ENV}, access it via:"
        echo "  spack env activate -p ${LBANN_ENV}"
    fi
fi

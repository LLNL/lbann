#!/bin/sh

SPACK_VERSION=$(spack --version)
MIN_SPACK_VERSION=0.13.3

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
    if [[ ${CORI} -eq 1 ]]; then
        CENTER="nersc"
        # Make sure to purge and setup the modules properly prior to finding the Spack architecture
        source ${SPACK_ENV_DIR}/${CENTER}/setup_modules.sh
    else
        CENTER="llnl_lc"
    fi
fi

SPACK_ARCH=$(spack arch)
SPACK_ARCH_TARGET=$(spack arch -t)

SCRIPT=$(basename ${BASH_SOURCE})
BUILD_DIR=${LBANN_HOME}/build/spack
ENABLE_GPUS="+gpu+nccl"
BUILD_TYPE=Release
VERBOSE=0
LBANN_ENV=lbann-${SPACK_ARCH_TARGET}

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
  ${C}-e | --env${N}           Build and install LBANN in the spack environment provided.
  ${C}--disable-gpus${N}       Disable GPUS
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
        --disable-gpus)
            ENABLE_GPUS=
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

SPACK_ENV=$(cat <<EOF
spack:
  concretization: together
  specs:
  - lbann@develop${ENABLE_GPUS}
  packages:
${EXTERNAL_ALL_PACKAGES}
${COMPILER_ALL_PACKAGES}

${EXTERNAL_PACKAGES}

${STD_PACKAGES}

    aluminum:
      buildable: true
      version: [master]
      providers: {}
      paths: {}
      modules: {}
      compiler: []
      target: []
    hydrogen:
      buildable: true
      version: [develop]
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
echo ${CMD}
${CMD}

CMD="spack env activate -p ${LBANN_ENV}"
echo ${CMD}
${CMD}

CMD="spack install"
echo ${CMD}
eval ${CMD}
if [[ $? -ne 0 ]]; then
    echo "--------------------"
    echo "Spack installation FAILED"
    echo "--------------------"
    exit 1
else
    echo "LBANN is installed in a spack environment named ${LBANN_ENV}, access it via:"
    echo "  spack env activate -p ${LBANN_ENV}"
fi

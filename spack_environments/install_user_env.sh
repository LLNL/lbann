#!/bin/sh

SPACK_VERSION=$(spack --version)
MIN_SPACK_VERSION=0.13.3

compare_versions()
{
    local v1=( $(echo "$1" | tr '.' ' ') )
    local v2=( $(echo "$2" | tr '.' ' ') )
    local len=$(( ${#v1[*]} > ${#v2[*]} ? ${#v1[*]} : ${#v2[*]} ))
    #local len=3
    for ((i=0; i<len; i++))
    do
        [ "${v1[i]:-0}" -gt "${v2[i]:-0}" ] && return 1
        [ "${v1[i]:-0}" -lt "${v2[i]:-0}" ] && return 2
    done
    return 0
}

compare_versions ${SPACK_VERSION} ${MIN_SPACK_VERSION}
VALID_SPACK=$?

if [[ ${VALID_SPACK} -eq 2 ]]; then
    echo "Newer version of Spack required.  Detected version ${SPACK_VERSION} requires at least ${MIN_SPACK_VERSION}"
    exit 1
fi

SPACK_ENV_DIR=$(dirname ${BASH_SOURCE})

if [[ ${SPACK_ENV_DIR} = "." ]]; then
    LBANN_HOME=$(dirname ${PWD})
    SPACK_ENV_DIR=${LBANN_HOME}/spack_environments
elif [[ ${SPACK_ENV_DIR} = /* ]]; then
    LBANN_HOME=$(dirname ${SPACK_ENV_DIR})
else
    LBANN_HOME=$(dirname ${PWD}/${SPACK_ENV_DIR})
    SPACK_ENV_DIR=${LBANN_HOME}/spack_environments
fi

# Detect system parameters
CLUSTER=$(hostname | sed 's/\([a-zA-Z][a-zA-Z]*\)[0-9]*/\1/g')
ARCH=$(uname -m)
SYS=$(uname -s)

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

SCRIPT=$(basename ${BASH_SOURCE})
BUILD_DIR=${LBANN_HOME}/build/spack
ENABLE_GPUS="+gpu+nccl"
BUILD_ENV=TRUE
BUILD_TYPE=Release
VERBOSE=0
LBANN_ENV=lbann

if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    echo "script ${BASH_SOURCE[0]} is being sourced ... only setting environment variables."
    BUILD_ENV="FALSE"
fi

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
            return
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

spack env create ${LBANN_ENV} ${temp_file}
spack env activate -p ${LBANN_ENV}
spack install

echo "LBANN is installed in a spack environment named ${LBANN_ENV}, access it via:"
echo "  spack env activate -p ${LBANN_ENV}"

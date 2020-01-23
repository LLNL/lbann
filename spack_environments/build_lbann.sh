#!/bin/sh

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
SPACK_ARCH=$(spack arch)
SYS=$(uname -s)

SCRIPT=$(basename ${BASH_SOURCE})
BUILD_DIR=${LBANN_HOME}/build/spack
ENABLE_GPUS=ON
BUILD_ENV=TRUE
BUILD_TYPE=Release

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
  ${C}--debug${N}              Build with debug flag.
  ${C}--verbose${N}            Verbose output.
  ${C}-p | --prefix${N}        Build and install LBANN headers and dynamic library into subdirectorys at this path prefix.
  ${C}-i | --install-dir${N}   Install LBANN headers and dynamic library into the install directory.
  ${C}-b | --build-dir${N}     Specify alternative build directory; default is <lbann_home>/build/spack.
  ${C}--disable-gpus${N}       Disable GPUS
  ${C}-r | --rebuild-env${N}   Rebuild the environment variables and load the modules
  ${C}--instrument${N}         Use -finstrument-functions flag, for profiling stack traces
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
        -b|--build-dir)
            # Change default build directory
            if [ -n "${2}" ]; then
                if [[ ${2} = "." ]]; then
                    BUILD_DIR=${SPACK_ENV_DIR}/${2}
                elif [[ ${2} = /* ]]; then
                    BUILD_DIR=${2}
                else
                    BUILD_DIR=${SPACK_ENV_DIR}/${2}
                fi
                shift
            else
                echo "\"${1}\" option requires a non-empty option argument" >&2
                exit 1
            fi
            ;;
        -i|--install-dir)
            # Specify install directory
            if [ -n "${2}" ]; then
                if [[ ${2} = "." ]]; then
                    INSTALL_DIR=${SPACK_ENV_DIR}/${2}
                elif [[ ${2} = /* ]]; then
                    INSTALL_DIR=${2}
                else
                    INSTALL_DIR=${SPACK_ENV_DIR}/${2}
                fi
                shift
            else
                echo "\"${1}\" option requires a non-empty option argument" >&2
                exit 1
            fi
            ;;
        -p|--prefix)
            # Change default build directory
            if [ -n "${2}" ]; then
                if [[ ${2} = "." ]]; then
                    BUILD_DIR=${SPACK_ENV_DIR}/${2}/build
                    INSTALL_DIR=${SPACK_ENV_DIR}/${2}/install
                elif [[ ${2} = /* ]]; then
                    BUILD_DIR=${2}/build
                    INSTALL_DIR=${2}/install
                else
                    BUILD_DIR=${SPACK_ENV_DIR}/${2}/build
                    INSTALL_DIR=${SPACK_ENV_DIR}/${2}/install
                fi
                shift
            else
                echo "\"${1}\" option requires a non-empty option argument" >&2
                exit 1
            fi
            ;;
        --disable-gpus)
            ENABLE_GPUS=OFF
            ;;
        -r|--rebuild-env)
            BUILD_ENV=FALSE
            ;;
        -v|--verbose)
            # Verbose output
            VERBOSE=1
            ;;
        -d|--debug)
            # Debug mode
            BUILD_TYPE=Debug
            DETERMINISTIC=ON
            ;;
        --instrument)
            INSTRUMENT="-finstrument-functions -ldl"
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

INSTALL_DIR="${INSTALL_DIR:-${LBANN_HOME}/build/gnu.${BUILD_TYPE}.${CLUSTER}.llnl.gov}"

export LBANN_HOME=${LBANN_HOME}
export LBANN_BUILD_DIR=${BUILD_DIR}
export LBANN_INSTALL_DIR=${INSTALL_DIR}

if [[ ${BUILD_ENV} == "FALSE" ]]; then
    echo "Setting environment variables: "
    echo "LBANN_HOME=${LBANN_HOME}"
    echo "LBANN_BUILD_DIR=${LBANN_BUILD_DIR}"
    echo "LBANN_INSTALL_DIR=${LBANN_INSTALL_DIR}"
    CMD="source ${LBANN_BUILD_DIR}/loads"
    echo ${CMD}
    ${CMD}
    return
fi

CMD="mkdir -p ${BUILD_DIR}"
echo ${CMD}
${CMD}
CMD="mkdir -p ${INSTALL_DIR}"
echo ${CMD}
${CMD}

CENTER=
SPACK_ENV=
SUPERBUILD=
if [[ ${SYS} = "Darwin" ]]; then
    OSX_VER=$(sw_vers -productVersion)
    SPACK_ENV=developer_release_osx_spack.yaml
    SUPERBUILD=superbuild_lbann_osx.sh
else
    CORI=$([[ $(hostname) =~ (cori|cgpu) ]] && echo 1 || echo 0)
    if [[ ${CORI} -eq 1 ]]; then
	CENTER="nersc"
    else
	CENTER="llnl_lc"
    fi
    if [[ "${DISABLE_GPUS}" == "ON" ]]; then
        SPACK_ENV=developer_release_spack.yaml
    else
        SPACK_ENV=developer_release_cuda_spack.yaml
    fi
    SUPERBUILD=superbuild_lbann.sh
fi

source ${SPACK_ENV_DIR}/setup_lbann_dependencies.sh

if [[ ${SYS} = "Darwin" ]]; then
    export DYLD_LIBRARY_PATH=/System/Library/Frameworks/ImageIO.framework/Resources/:/usr/lib/:${DYLD_LIBRARY_PATH}
fi

C_FLAGS="${INSTRUMENT} -fno-omit-frame-pointer"
CXX_FLAGS="-DLBANN_SET_EL_RNG -mcpu=native ${INSTRUMENT} -fno-omit-frame-pointer"

if [[ ${ARCH} = "x86_64" ]]; then
    CXX_FLAGS="-march=native ${CXX_FLAGS}"
else
    CXX_FLAGS="-mcpu=native ${CXX_FLAGS}"
fi

source ${SPACK_ENV_DIR}/${SUPERBUILD}

ninja

#!/bin/bash

# "spack" is just a shell function; it may not be exported to this
# scope. Just to be sure, reload the shell integration.
if [ -n "${SPACK_ROOT}" ]; then
    source ${SPACK_ROOT}/share/spack/setup-env.sh
fi

SPACK_VERSION=$(spack --version | sed 's/-.*//g')
MIN_SPACK_VERSION=0.16.0

source $(dirname ${BASH_SOURCE})/utilities.sh

compare_versions ${SPACK_VERSION} ${MIN_SPACK_VERSION}
VALID_SPACK=$?

if [[ ${VALID_SPACK} -eq 2 ]]; then
    echo "Newer version of Spack required.  Detected version ${SPACK_VERSION} requires at least ${MIN_SPACK_VERSION}"
    exit 1
fi

# Detect system parameters
CLUSTER=$(hostname | sed 's/\([a-zA-Z][a-zA-Z]*\)[0-9]*/\1/g')
#ARCH=$(uname -m)
SYS=$(uname -s)

SCRIPT=${BASH_SOURCE}

if [[ ${SYS} = "Darwin" ]]; then
SCRIPTS_DIR=$(normpath $(dirname $(osx_realpath ${SCRIPT})))
else
SCRIPTS_DIR=$(realpath $(dirname ${SCRIPT}))
fi

LBANN_HOME=$(dirname ${SCRIPTS_DIR})
#SPACK_ENV_DIR=${LBANN_HOME}/spack_environments

# Identify the center that we are running at
CENTER=
# Customize the build based on the center
source $(dirname ${BASH_SOURCE})/customize_build_env.sh
set_center_specific_fields 

# Temporarily overwrite the build suffix
BUILD_SUFFIX=

SPACK_ARCH=$(spack arch)
SPACK_ARCH_TARGET=$(spack arch -t)
SPACK_ARCH_PLATFORM=$(spack arch -p)
SPACK_ARCH_GENERIC_TARGET=$(spack python -c "import archspec.cpu as cpu; print(str(cpu.host().family))")
# Create a modified spack arch with generic target architecture
SPACK_ARCH_PLATFORM_GENERIC_TARGET="${SPACK_ARCH_PLATFORM}-${SPACK_ARCH_GENERIC_TARGET}"

SCRIPT=$(basename ${BASH_SOURCE})
#BUILD_DIR=${LBANN_HOME}/build/spack
LBANN_ENV=
#SPACK_INSTALL_ARGS=
#MERLIN_PACKAGES=

INSTALL_DEPS=

DRY_RUN=
# Flag for passing subcommands to spack dev-build
DEV_BUILD_FLAGS=

LBANN_LABEL="local"
LBANN_VARIANTS=

# Default versions of Hydrogen, DiHydrogen, and Aluminum - use head of repo
HYDROGEN_VER="@develop"
ALUMINUM_VER="@master"
DIHYDROGEN_VER="@develop"

# Define the GPU_ARCH_VARIANTS field
GPU_ARCH_VARIANTS=
set_center_specific_gpu_arch ${CENTER} ${SPACK_ARCH_TARGET}
GPU_VARIANTS+=" ${GPU_ARCH_VARIANTS}"

################################################################
# Help message
################################################################

function help_message {
    local SCRIPT=$(basename ${BASH_SOURCE})
    local N=$(tput sgr0)    # Normal text
    local C=$(tput setf 4)  # Colored text
    cat << EOF
######################################################################################
Build LBANN: has preconfigured module lists for LLNL LC, OLCF, and NERSC systems.
  Will create a build directory with the name spack-build-<hash> in the root of the LBANN project tree.
  primary variants that can be passed into lbann come from spack and can be seen with:
    spack info lbann
  and passed to this script via:
    ${SCRIPT} -- <variants>
######################################################################################
Usage: ${SCRIPT} [options] -- [list of spack variants]
Options:
  ${C}--help${N}                  Display this help message and exit.
  ${C}--build-env-only SHELL${N}  Drop into a shell with all of the spack build environment setup
  ${C}--build-suffix SUFFIX${N}   Appends the string to the sym link pointing to the build directory
  ${C}--config-only${N}           Run the spack dev-build command up to the configure stage only
  ${C}-d | --deps-only)${N}       Only install the lbann dependencies
  ${C}--drop-in SHELL${N}         Drop into a shell with all of the spack build environment setup after setting up the dev-build
  ${C}--dry-run${N}               Dry run the commands (no effect)
  ${C}-e | --env ENV${N}          Build and install LBANN in the spack environment provided
  ${C}-l | --label${N}            LBANN local version label.
  ${C}--no-modules${N}            Don't try to load any modules (use the existing users environment)
  ${C}-s | --stable${N}           Use the latest stable defaults not the head of Hydrogen, DiHydrogen and Aluminum repos
  ${C}--${N}                      Pass all variants to spack after the dash dash (--)
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
        --build-env-only)
            BUILD_ENV_ONLY="TRUE"
            if [ -n "${2}" ]; then
                BUILD_ENV_SHELL="${2}"
                shift
            else
                echo "\"${1}\" option requires a non-empty option argument" >&2
                exit 1
            fi
            ;;
        --build-suffix)
            if [ -n "${2}" ]; then
                BUILD_SUFFIX=".${2}"
                shift
            else
                echo "\"${1}\" option requires a non-empty option argument" >&2
                exit 1
            fi
            ;;
        --config-only)
            DEV_BUILD_FLAGS+=" -u cmake"
            ;;
        -d|--install-deps)
            INSTALL_DEPS="TRUE"
            ;;
        --drop-in)
            if [ -n "${2}" ]; then
                DEV_BUILD_FLAGS+=" --drop-in ${2}"
                shift
            else
                echo "\"${1}\" option requires a non-empty option argument" >&2
                exit 1
            fi
            ;;
        --dry-run)
            DRY_RUN="TRUE"
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
        -l|--label)
            # Change default LBANN version label
            if [ -n "${2}" ]; then
                LBANN_LABEL=${2}
                shift
            else
                echo "\"${1}\" option requires a non-empty option argument" >&2
                exit 1
            fi
            ;;
        --no-modules)
            SKIP_MODULES="TRUE"
            ;;
        -s|--stable-defaults)
            # Use the latest released version
            HYDROGEN_VER=
            ALUMINUM_VER=
            DIHYDROGEN_VER=
            ;;
        --)
            shift
            LBANN_VARIANTS=${*}
            break
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

LBANN_ENV="${LBANN_ENV:-lbann-${LBANN_LABEL}-${SPACK_ARCH_TARGET}}"
CORE_BUILD_PATH="${LBANN_HOME}/build/${CLUSTER}.${LBANN_ENV}${BUILD_SUFFIX:-}"

LOG="spack-build-${LBANN_ENV}.log"
rm ${LOG}

if [[ ! "${LBANN_VARIANTS}" =~ .*"^hydrogen".* ]]; then
    # If the user didn't supply a specific version of Hydrogen on the command line add one
    HYDROGEN="^hydrogen${HYDROGEN_VER}+al"
fi

if [[ ! "${LBANN_VARIANTS}" =~ .*"^aluminum".* ]]; then
    # If the user didn't supply a specific version of Aluminum on the command line add one
    ALUMINUM="^aluminum${ALUMINUM_VER}"
fi

# Check to see if LBANN turned on dihydrogen
DIHYDROGEN_VARIANTS='+dihydrogen'
if [[ "${LBANN_VARIANTS}" =~ .*"${DIHYDROGEN_VARIANTS}".* ]]; then
    if [[ ! "${LBANN_VARIANTS}" =~ .*"^dihydrogen".* ]]; then
        # If the user didn't supply a specific version of DiHydrogen on the command line add one
        DIHYDROGEN="^dihydrogen${DIHYDROGEN_VER}"
    fi
fi

GPU_VARIANTS='+cuda'
if [[ "${LBANN_VARIANTS}" =~ .*"${GPU_VARIANTS}".* ]]; then
    # Prepend the GPU_ARCH_Variants for the LBANN variants
    LBANN_VARIANTS=" ${GPU_ARCH_VARIANTS} ${LBANN_VARIANTS}"
fi

# Establish the spec for LBANN
LBANN_SPEC="lbann@${LBANN_LABEL}${LBANN_VARIANTS} ${HYDROGEN} ${DIHYDROGEN} ${ALUMINUM} ${MPI}"

if [[ ! -n "${SKIP_MODULES:-}" ]]; then
    # Activate modules
    MODULE_CMD=
    set_center_specific_modules ${CENTER} ${SPACK_ARCH_TARGET}
    echo ${MODULE_CMD} | tee -a ${LOG}
    [[ -z "${DRY_RUN:-}" ]] && eval ${MODULE_CMD}
fi

#LBANN_FIND_CMD="spack find --format {hash:7} lbann@${LBANN_LABEL} arch=${SPACK_ARCH}"
#echo ${LBANN_FIND_CMD}
#LBANN_HASH=$(${LBANN_FIND_CMD})
#if [[ -n "${LBANN_HASH}" && ! "${LBANN_HASH}" == "No package matches the query" ]]; then
#    echo "Unable to find the package"
#else
#    CMD="spack uninstall -y -a lbann@${LBANN_LABEL} arch=${SPACK_ARCH}"
#    echo ${CMD}
#    ${CMD}
#fi
#echo "Found the hash ${LBANN_FIND_CMD}"

if [[ -n "${INSTALL_DEPS:-}" ]]; then
    if [[ $(spack env list | grep ${LBANN_ENV}) ]]; then
        echo "Spack environment ${LBANN_ENV} already exists... overwriting it"
        CMD="spack env rm --yes-to-all ${LBANN_ENV}"
        echo ${CMD} | tee -a ${LOG}
        [[ -z "${DRY_RUN:-}" && -n "${INSTALL_DEPS:-}" ]] && ${CMD}
    fi

    CMD="spack env create ${LBANN_ENV}"
    echo ${CMD} | tee -a ${LOG}
    [[ -z "${DRY_RUN:-}" ]] && ${CMD}
fi

CMD="spack env activate -p ${LBANN_ENV}"
echo ${CMD} | tee -a ${LOG}
[[ -z "${DRY_RUN:-}" ]] && ${CMD}

if [[ -n "${BUILD_ENV_ONLY:-}" ]]; then
    CMD="spack build-env ${LBANN_SPEC} -- ${BUILD_ENV_SHELL}"
    echo ${CMD} | tee -a ${LOG}
    [[ -z "${DRY_RUN:-}" ]] && ${CMD}
    exit
fi

if [[ -n "${INSTALL_DEPS:-}" ]]; then
    CMD="spack external find --scope env:${LBANN_ENV}"
    echo ${CMD} | tee -a ${LOG}
    [[ -z "${DRY_RUN:-}" ]] && ${CMD}
fi

# Figure out if there is a default MPI library for the center
MPI=
set_center_specific_mpi ${CENTER} ${SPACK_ARCH_TARGET}

CMD="spack spec ${LBANN_SPEC}"
echo ${CMD} | tee -a ${LOG}
if [[ -z "${DRY_RUN:-}" ]]; then
    eval ${CMD}
    if [[ $? -ne 0 ]]; then
        echo "-----------------"
        echo "Spack spec FAILED"
        echo "-----------------"
        exit 1
    fi
fi

CMD="spack install --only dependencies ${LBANN_SPEC}"
[[ -n "${INSTALL_DEPS:-}" ]] && echo ${CMD} | tee -a ${LOG}
if [[ -z "${DRY_RUN:-}" && -n "${INSTALL_DEPS:-}" ]]; then
    eval ${CMD}
    if [[ $? -ne 0 ]]; then
        echo "-----------------------------------------"
        echo "Spack installation of dependenceis FAILED"
        echo "-----------------------------------------"
        exit 1
    fi
fi

LINK_DIR="${LINK_DIR:-${CORE_BUILD_PATH}}"

BUILD_DIR=$(dirname ${LINK_DIR})

CMD="mkdir -p ${BUILD_DIR}"
echo ${CMD}
[[ -z "${DRY_RUN:-}" ]] && ${CMD}

CMD="spack dev-build --source-path ${LBANN_HOME} ${DEV_BUILD_FLAGS} ${LBANN_SPEC}"
echo ${CMD} | tee -a ${LOG}
[[ -z "${DRY_RUN:-}" ]] && ${CMD}

if [[ -n "${INSTALL_DEPS}" ]]; then
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
fi

LBANN_BUILD_DIR=$(grep "PROJECT_BINARY_DIR:" ${LBANN_HOME}/spack-build-out.txt | awk '{print $2}')

#LBANN_HASH=$(${LBANN_FIND_CMD})
#if [[ "${LBANN_HASH}" == "No package matches the query" ]]; then
#    echo "Unable to find the package"
#fi

if [[ -L "${LINK_DIR}" && -d "${LINK_DIR}" ]]; then
    CMD="rm ${LINK_DIR}"
    echo ${CMD} | tee -a ${LOG}
    [[ -z "${DRY_RUN:-}" ]] && ${CMD}
fi

CMD="ln -s ${LBANN_BUILD_DIR} ${LINK_DIR}"
echo ${CMD} | tee -a ${LOG}
[[ -z "${DRY_RUN:-}" ]] && ${CMD}
#exit

#    else
echo "LBANN is installed in a spack environment named ${LBANN_ENV}, access it via:" | tee -a ${LOG}
echo "  spack env activate -p ${LBANN_ENV}" | tee -a ${LOG}
echo "To rebuild LBANN from source drop into a shell with the spack build environment setup:" | tee -a ${LOG}
CMD="spack build-env ${LBANN_SPEC} -- bash"
echo ${CMD} | tee -a ${LOG}
echo "To use this version of LBANN have spack load it's module:is installed in a spack environment named ${LBANN_ENV}, access it via:" | tee -a ${LOG}
echo "spack load lbann@${LBANN_LABEL}" | tee -a ${LOG}

#echo "I dropped you into a bash shell in the build-env for the package:"
#echo "To uninstall execute:"

#    fi
#fi


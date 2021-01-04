#!/bin/bash

ORIG_CMD="$0 $@"
SCRIPT=${BASH_SOURCE}

if [[ ${SYS} = "Darwin" ]]; then
SCRIPTS_DIR=$(normpath $(dirname $(osx_realpath ${SCRIPT})))
else
SCRIPTS_DIR=$(realpath $(dirname ${SCRIPT}))
fi

LBANN_HOME=$(dirname ${SCRIPTS_DIR})
#SPACK_ENV_DIR=${LBANN_HOME}/spack_environments

SCRIPT=$(basename ${BASH_SOURCE})
LBANN_ENV=
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

################################################################
# Help message
################################################################

function help_message {
    local SCRIPT=$(basename ${BASH_SOURCE})
    local N=$(tput sgr0)    # Normal text
    local C=$(tput setf 4)  # Colored text
    cat << EOF
##########################################################################################
Build LBANN: has preconfigured module lists for LLNL LC, OLCF, and NERSC systems.
  Will create a build directory with the name spack-build-<hash> in the root of the LBANN project tree.
  primary variants that can be passed into lbann come from spack and can be seen with:
    spack info lbann
  and passed to this script via:
    ${SCRIPT} -- <variants>
##########################################################################################
Usage: ${SCRIPT} [options] -- [list of spack variants]
Options:
  ${C}--help${N}                  Display this help message and exit.
  ${C}--build-env-only SHELL${N}  Drop into a shell with all of the spack build environment setup
  ${C}--build-suffix SUFFIX${N}   Appends the string to the sym link pointing to the build directory
  ${C}--config-only${N}           Run the spack dev-build command up to the configure stage only
  ${C}-d | --install-deps${N}    Install the lbann dependencies in addition to building from local source
  ${C}--dependencies-only${N}    Stop after installing the lbann dependencies
  ${C}--drop-in SHELL${N}         Drop into a shell with all of the spack build environment setup after setting up the dev-build
  ${C}--dry-run${N}               Dry run the commands (no effect)
  ${C}-e | --env ENV${N}          Build and install LBANN in the spack environment provided
  ${C}-l | --label${N}            LBANN local version label.
  ${C}--no-modules${N}            Don't try to load any modules (use the existing users environment)
  ${C}--spec-only${N}             Stop after a spack spec command
  ${C}-s | --stable${N}           Use the latest stable defaults not the head of Hydrogen, DiHydrogen and Aluminum repos
  ${C}--hydrogen-repo PATH${N}    Use a local repository for the Hydrogen library
  ${C}--dihydrogen-repo PATH${N}  Use a local repository for the DiHydrogen library
  ${C}--aluminum-repo PATH${N}    Use a local repository for the Aluminum library
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
        --dependencies-only)
            DEPENDENCIES_ONLY="TRUE"
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
        --spec-only)
            SPEC_ONLY="TRUE"
            ;;
        -s|--stable-defaults)
            # Use the latest released version
            HYDROGEN_VER=
            ALUMINUM_VER=
            DIHYDROGEN_VER=
            ;;
        --hydrogen-repo)
            if [ -n "${2}" ]; then
                HYDROGEN_PATH=${2}
                shift
            else
                echo "\"${1}\" option requires a non-empty option argument" >&2
                exit 1
            fi
            ;;
        --dihydrogen-repo)
            if [ -n "${2}" ]; then
                DIHYDROGEN_PATH=${2}
                shift
            else
                echo "\"${1}\" option requires a non-empty option argument" >&2
                exit 1
            fi
            ;;
        --aluminum-repo)
            if [ -n "${2}" ]; then
                ALUMINUM_PATH=${2}
                shift
            else
                echo "\"${1}\" option requires a non-empty option argument" >&2
                exit 1
            fi
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

# "spack" is just a shell function; it may not be exported to this
# scope. Just to be sure, reload the shell integration.
if [ -n "${SPACK_ROOT}" ]; then
    source ${SPACK_ROOT}/share/spack/setup-env.sh
else
    echo "Spack required.  Please set SPACK_ROOT environment variable"
    exit 1
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

# Define the GPU_ARCH_VARIANTS field
GPU_ARCH_VARIANTS=
set_center_specific_gpu_arch ${CENTER} ${SPACK_ARCH_TARGET}
GPU_VARIANTS+=" ${GPU_ARCH_VARIANTS}"

LBANN_ENV="${LBANN_ENV:-lbann-${LBANN_LABEL}-${SPACK_ARCH_TARGET}}"
CORE_BUILD_PATH="${LBANN_HOME}/build/${CLUSTER}.${LBANN_ENV}${BUILD_SUFFIX:-}"

LOG="spack-build-${LBANN_ENV}.log"
if [[ -f ${LOG} ]]; then
    CMD="rm ${LOG}"
    echo ${CMD}
    [[ -z "${DRY_RUN:-}" ]] && ${CMD}
fi

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

# Record the original command in the log file
echo "${ORIG_CMD}" | tee -a ${LOG}

# Uninstall any existing versions for this architecture with the same label
LBANN_FIND_CMD="spack find --format {hash:7} lbann@${LBANN_LABEL} arch=${SPACK_ARCH}"
echo ${LBANN_FIND_CMD} | tee -a ${LOG}
LBANN_HASH=$(${LBANN_FIND_CMD})
if [[ -n "${LBANN_HASH}" && ! "${LBANN_HASH}" =~ "No package matches the query" ]]; then
    LBANN_HASH_ARRAY=(${LBANN_HASH})
    for h in ${LBANN_HASH_ARRAY[@]}
    do
        CMD="spack uninstall -y lbann@${LBANN_LABEL} arch=${SPACK_ARCH} /${h}"
        echo ${CMD} | tee -a ${LOG}
        [[ -z "${DRY_RUN:-}" ]] && ${CMD}
    done
fi

if [[ ! -n "${SKIP_MODULES:-}" ]]; then
    # Activate modules
    MODULE_CMD=
    set_center_specific_modules ${CENTER} ${SPACK_ARCH_TARGET}
    if [[ -n ${MODULE_CMD} ]]; then
        echo ${MODULE_CMD} | tee -a ${LOG}
        [[ -z "${DRY_RUN:-}" ]] && eval ${MODULE_CMD}
    fi
fi

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
if [[ -z "${DRY_RUN:-}" ]]; then
    if [[ -z $(spack env list | grep ${LBANN_ENV}) ]]; then
        echo "Spack could not activate environment ${LBANN_ENV} -- install dependencies with -d flag"
        exit 1
    fi
    ${CMD}
fi

# Figure out if there is a default MPI library for the center
MPI=
set_center_specific_mpi ${CENTER} ${SPACK_ARCH_TARGET}

##########################################################################################
# Establish the spec for LBANN
LBANN_SPEC="lbann@${LBANN_LABEL}${LBANN_VARIANTS} ${HYDROGEN} ${DIHYDROGEN} ${ALUMINUM} ${MPI}"
LBANN_DEV_PATH_SPEC="lbann@${LBANN_LABEL}${LBANN_VARIANTS} dev_path=${LBANN_HOME} ${HYDROGEN} ${DIHYDROGEN} ${ALUMINUM} ${MPI}"
##########################################################################################

if [[ -n "${BUILD_ENV_ONLY:-}" ]]; then
    CMD="spack build-env ${LBANN_SPEC} -- ${BUILD_ENV_SHELL}"
    echo ${CMD} | tee -a ${LOG}
    [[ -z "${DRY_RUN:-}" ]] && ${CMD}
    exit
fi

if [[ -n "${INSTALL_DEPS:-}" ]]; then
    CMD="spack compiler find --scope env:${LBANN_ENV}"
    echo ${CMD} | tee -a ${LOG}
    [[ -z "${DRY_RUN:-}" ]] && ${CMD}
    CMD="spack external find --scope env:${LBANN_ENV}"
    echo ${CMD} | tee -a ${LOG}
    [[ -z "${DRY_RUN:-}" ]] && ${CMD}

    # See if there are any center-specific externals
    SPACK_ENV_YAML_FILE="${SPACK_ROOT}/var/spack/environments/${LBANN_ENV}/spack.yaml"
    CMD="set_center_specific_externals ${CENTER} ${SPACK_ARCH_TARGET} ${SPACK_ENV_YAML_FILE}"
    echo ${CMD} | tee -a ${LOG}
    [[ -z "${DRY_RUN:-}" ]] && ${CMD}
fi

CMD="spack spec -l ${LBANN_DEV_PATH_SPEC}"
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
# Currently unused, but here is how to get the spack hash before dev-build is called
# LBANN_SPEC_HASH=$(spack spec -l ${LBANN_DEV_PATH_SPEC} | grep lbann | grep arch=${SPACK_ARCH} | awk '{print $1}')
[[ -z "${DRY_RUN:-}" && "${SPEC_ONLY}" == "TRUE" ]] && exit

##########################################################################################
# Tell the spack environment to use a local repository for these libraries
if [[ -n "${HYDROGEN_PATH:-}" ]]; then
    CMD="spack develop --no-clone -p ${HYDROGEN_PATH} hydrogen${HYDROGEN_VER}"
    echo "${CMD}" | tee -a ${LOG}
    ${CMD}
fi

if [[ -n "${DIHYDROGEN_PATH:-}" ]]; then
    CMD="spack develop --no-clone -p ${DIHYDROGEN_PATH} dihydrogen${DIHYDROGEN_VER}"
    echo "${CMD}" | tee -a ${LOG}
    ${CMD}
fi

if [[ -n "${ALUMINUM_PATH:-}" ]]; then
    CMD="spack develop --no-clone -p ${ALUMINUM_PATH} aluminum${ALUMINUM_VER}"
    echo "${CMD}" | tee -a ${LOG}
    ${CMD}
fi
##########################################################################################

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
    if [[ -n "${DEPENDENCIES_ONLY:-}" ]]; then
        exit
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

LBANN_BUILD_DIR=$(grep "PROJECT_BINARY_DIR:" ${LBANN_HOME}/spack-build-out.txt | awk '{print $2}')

if [[ -L "${LINK_DIR}" && -d "${LINK_DIR}" ]]; then
    CMD="rm ${LINK_DIR}"
    echo ${CMD} | tee -a ${LOG}
    [[ -z "${DRY_RUN:-}" ]] && ${CMD}
fi

CMD="ln -s ${LBANN_BUILD_DIR} ${LINK_DIR}"
echo ${CMD} | tee -a ${LOG}
[[ -z "${DRY_RUN:-}" ]] && ${CMD}

echo "##########################################################################################" | tee -a ${LOG}
echo "LBANN is installed in a spack environment named ${LBANN_ENV}, access it via:" | tee -a ${LOG}
echo "  spack env activate -p ${LBANN_ENV}" | tee -a ${LOG}
echo "To rebuild LBANN from source drop into a shell with the spack build environment setup:" | tee -a ${LOG}
echo "  spack build-env ${LBANN_SPEC} -- bash" | tee -a ${LOG}
echo "To use this version of LBANN have spack load it's module:is installed in a spack environment named ${LBANN_ENV}, access it via:" | tee -a ${LOG}
echo "  spack load lbann@${LBANN_LABEL} arch=${SPACK_ARCH}" | tee -a ${LOG}
echo "##########################################################################################" | tee -a ${LOG}

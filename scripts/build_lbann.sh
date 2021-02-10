#!/bin/bash

ORIG_CMD="$0 $@"
SCRIPT=${BASH_SOURCE}

# Grab some helper functions
source $(dirname ${BASH_SOURCE})/utilities.sh

# Detect system parameters
SYS=$(uname -s)
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
CLEAN_BUILD=
# Flag for passing subcommands to spack dev-build
DEV_BUILD_FLAGS=
# Flag for passing subcommands to spack install and dev-build
INSTALL_DEV_BUILD_EXTRAS=

LBANN_VARIANTS=
CMD_LINE_VARIANTS=

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
  ${C}--clean-build${N}           Delete the local link to the build directory
  ${C}--clean-deps${N}            Forcibly uninstall Hydrogen, Aluminum, and DiHydrogen dependencies
  ${C}--config-only${N}           Run the spack dev-build command up to the configure stage only
  ${C}-d | --install-deps${N}     Install the lbann dependencies in addition to building from local source
  ${C}--dependencies-only${N}     Stop after installing the lbann dependencies
  ${C}--drop-in SHELL${N}         Drop into a shell with all of the spack build environment setup after setting up the dev-build
  ${C}--dry-run${N}               Dry run the commands (no effect)
  ${C}-l | --label${N}            LBANN version label prefix: (default label is local-<SPACK_ARCH_TARGET>,
                          and is built and installed in the spack environment lbann-<label>-<SPACK_ARCH_TARGET>
  ${C}--no-modules${N}            Don't try to load any modules (use the existing users environment)
  ${C}--no-tmp-build-dir${N}      Don't put the build directory in tmp space
  ${C}--spec-only${N}             Stop after a spack spec command
  ${C}-s | --stable${N}           Use the latest stable defaults not the head of Hydrogen, DiHydrogen and Aluminum repos
  ${C}--test${N}                  Enable local unit tests
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
        --clean-build)
            CLEAN_BUILD="TRUE"
            ;;
        --clean-deps)
            CLEAN_DEPS="TRUE"
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
        -l|--label)
            # Change default LBANN version label
            if [ -n "${2}" ]; then
                LBANN_LABEL_PREFIX=${2}
                shift
            else
                echo "\"${1}\" option requires a non-empty option argument" >&2
                exit 1
            fi
            ;;
        --no-modules)
            SKIP_MODULES="TRUE"
            ;;
        --no-tmp-build-dir)
            CLEAN_BUILD="TRUE"
            NO_TMP_BUILD_DIR="TRUE"
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
        --test)
            INSTALL_DEV_BUILD_EXTRAS="--test root"
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
            CMD_LINE_VARIANTS=${*}
            LBANN_VARIANTS=${CMD_LINE_VARIANTS}
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

function exit_on_failure()
{
    local cmd="$1"
    echo "FAILED: ${cmd}"
    exit 1
}

function uninstall_specific_versions()
{
    local package="$1"
    local version="$2"

    SPACK_ARCH=$(spack arch)
    # Ensure that only versions for this architecture are found
    FIND_CMD="spack find --format {hash:7} ${package}${version} arch=${SPACK_ARCH}"
    echo ${FIND_CMD} | tee -a ${LOG}
    HASH=$(${FIND_CMD})
    if [[ -n "${HASH}" && ! "${HASH}" =~ "No package matches the query" ]]; then
        HASH_ARRAY=(${HASH})
        for h in ${HASH_ARRAY[@]}
        do
            CMD="spack uninstall -y --force ${package}${version} /${h}"
            echo ${CMD} | tee -a ${LOG}
            [[ -z "${DRY_RUN:-}" ]] && (${CMD} || exit_on_failure "${CMD}")
        done
    fi
}

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

compare_versions ${SPACK_VERSION} ${MIN_SPACK_VERSION}
VALID_SPACK=$?

if [[ ${VALID_SPACK} -eq 2 ]]; then
    echo "Newer version of Spack required.  Detected version ${SPACK_VERSION} requires at least ${MIN_SPACK_VERSION}"
    exit 1
fi

# Detect system parameters
CLUSTER=$(hostname | sed 's/\([a-zA-Z][a-zA-Z]*\)[0-9]*/\1/g')

# Identify the center that we are running at
CENTER=
# Customize the build based on the center
source $(dirname ${BASH_SOURCE})/customize_build_env.sh
set_center_specific_fields

SPACK_ARCH=$(spack arch)
SPACK_ARCH_TARGET=$(spack arch -t)
SPACK_ARCH_PLATFORM=$(spack arch -p)
SPACK_ARCH_GENERIC_TARGET=$(spack python -c "import archspec.cpu as cpu; print(str(cpu.host().family))")
# Create a modified spack arch with generic target architecture
SPACK_ARCH_PLATFORM_GENERIC_TARGET="${SPACK_ARCH_PLATFORM}-${SPACK_ARCH_GENERIC_TARGET}"

LBANN_LABEL="${LBANN_LABEL_PREFIX:-local}-${SPACK_ARCH_TARGET}"
LBANN_ENV="${LBANN_ENV:-lbann-${LBANN_LABEL}}"
CORE_BUILD_PATH="${LBANN_HOME}/build/${CLUSTER}.${LBANN_ENV}"

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
    # Define the GPU_ARCH_VARIANTS field
    GPU_ARCH_VARIANTS=
    set_center_specific_gpu_arch ${CENTER} ${SPACK_ARCH_TARGET}
    # Prepend the GPU_ARCH_VARIANTS for the LBANN variants if the +cuda variant is defined
    LBANN_VARIANTS=" ${GPU_ARCH_VARIANTS} ${LBANN_VARIANTS}"
fi

# Record the original command in the log file
echo "${ORIG_CMD}" | tee -a ${LOG}

if [[ ! -n "${SKIP_MODULES:-}" ]]; then
    # Activate modules
    MODULE_CMD=
    set_center_specific_modules ${CENTER} ${SPACK_ARCH_TARGET}
    if [[ -n ${MODULE_CMD} ]]; then
        echo ${MODULE_CMD} | tee -a ${LOG}
        [[ -z "${DRY_RUN:-}" ]] && { eval ${MODULE_CMD} || exit_on_failure "${MODULE_CMD}"; }
    fi
fi

# If the dependencies are being installed then you should clean things up
if [[ -n "${INSTALL_DEPS:-}" ]]; then
    # Remove any old environment with the same name
    if [[ $(spack env list | grep -e "${LBANN_ENV}$") ]]; then
        echo "Spack environment ${LBANN_ENV} already exists... overwriting it"
        CMD="spack env rm --yes-to-all ${LBANN_ENV}"
        echo ${CMD} | tee -a ${LOG}
        [[ -z "${DRY_RUN:-}" && -n "${INSTALL_DEPS:-}" ]] && { ${CMD} || exit_on_failure "${CMD}"; }
    fi

    # Create the environment
    CMD="spack env create ${LBANN_ENV}"
    echo ${CMD} | tee -a ${LOG}
    [[ -z "${DRY_RUN:-}" ]] && { ${CMD} || exit_on_failure "${CMD}"; }

fi

##########################################################################################
# If not just dropping into the build environment, uninstall any existing versions for this
# architecture with the same label -- note that this has to be done outside of an environment
if [[ -z "${BUILD_ENV_ONLY:-}" ]]; then
    # For finding the lbann version don't use the architecture because sometimes it is "downgraded"
    LBANN_FIND_CMD="spack find --format {hash:7} lbann@${LBANN_LABEL}"
    echo ${LBANN_FIND_CMD} | tee -a ${LOG}
    LBANN_HASH=$(${LBANN_FIND_CMD})
    if [[ -n "${LBANN_HASH}" && ! "${LBANN_HASH}" =~ "No package matches the query" ]]; then
        LBANN_HASH_ARRAY=(${LBANN_HASH})
        for h in ${LBANN_HASH_ARRAY[@]}
        do
            CMD="spack uninstall -y --force lbann@${LBANN_LABEL} /${h}"
            echo ${CMD} | tee -a ${LOG}
            [[ -z "${DRY_RUN:-}" ]] && { ${CMD} || exit_on_failure "${CMD}"; }
        done
    fi
fi

if [[ -n "${CLEAN_DEPS:-}" ]]; then
    uninstall_specific_versions "hydrogen" "${HYDROGEN_VER}"
    uninstall_specific_versions "aluminum" "${ALUMINUM_VER}"
    uninstall_specific_versions "dihydrogen" "${DIHYDROGEN_VER}"
fi

CMD="spack env activate -p ${LBANN_ENV}"
echo ${CMD} | tee -a ${LOG}
if [[ -z "${DRY_RUN:-}" ]]; then
    if [[ -z $(spack env list | grep -e "${LBANN_ENV}$") ]]; then
        echo "Spack could not activate environment ${LBANN_ENV} -- install dependencies with -d flag"
        exit 1
    fi
    ${CMD} || exit_on_failure "${CMD}"
fi

# Figure out if there is a default MPI library for the center
MPI=
set_center_specific_mpi ${CENTER} ${SPACK_ARCH_TARGET}

##########################################################################################
# Establish the spec for LBANN
LBANN_SPEC="lbann@${LBANN_LABEL}${LBANN_VARIANTS} ${HYDROGEN} ${DIHYDROGEN} ${ALUMINUM} ${MPI}"
LBANN_DEV_PATH_SPEC="lbann@${LBANN_LABEL} dev_path=${LBANN_HOME} ${LBANN_VARIANTS} ${HYDROGEN} ${DIHYDROGEN} ${ALUMINUM} ${MPI}"
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
    [[ -z "${DRY_RUN:-}" ]] && { ${CMD} || exit_on_failure "${CMD}"; }

    CMD="spack external find --scope env:${LBANN_ENV}"
    echo ${CMD} | tee -a ${LOG}
    [[ -z "${DRY_RUN:-}" ]] && { ${CMD} || exit_on_failure "${CMD}"; }

    # See if there are any center-specific externals
    SPACK_ENV_YAML_FILE="${SPACK_ROOT}/var/spack/environments/${LBANN_ENV}/spack.yaml"
    CMD="set_center_specific_externals ${CENTER} ${SPACK_ARCH_TARGET} ${SPACK_ARCH} ${SPACK_ENV_YAML_FILE}"
    echo ${CMD} | tee -a ${LOG}
    [[ -z "${DRY_RUN:-}" ]] && { ${CMD} || exit_on_failure "${CMD}"; }
fi

##########################################################################################
# Tell the spack environment to use a local repository for these libraries
if [[ -n "${HYDROGEN_PATH:-}" ]]; then
    CMD="spack develop --no-clone -p ${HYDROGEN_PATH} hydrogen${HYDROGEN_VER}"
    echo "${CMD}" | tee -a ${LOG}
    ${CMD} || exit_on_failure "${CMD}"
fi

if [[ -n "${DIHYDROGEN_PATH:-}" ]]; then
    CMD="spack develop --no-clone -p ${DIHYDROGEN_PATH} dihydrogen${DIHYDROGEN_VER}"
    echo "${CMD}" | tee -a ${LOG}
    ${CMD} || exit_on_failure "${CMD}"
fi

if [[ -n "${ALUMINUM_PATH:-}" ]]; then
    CMD="spack develop --no-clone -p ${ALUMINUM_PATH} aluminum${ALUMINUM_VER}"
    echo "${CMD}" | tee -a ${LOG}
    ${CMD} || exit_on_failure "${CMD}"
fi
##########################################################################################

CMD="spack spec -l ${LBANN_DEV_PATH_SPEC}"
echo ${CMD} | tee -a ${LOG}
if [[ -z "${DRY_RUN:-}" ]]; then
    eval ${CMD} || exit_on_failure "${CMD}"
fi
# Get the spack hash before dev-build is called
LBANN_SPEC_HASH=$(spack spec -l ${LBANN_DEV_PATH_SPEC} | grep lbann | grep arch=${SPACK_ARCH_PLATFORM} | awk '{print $1}')
[[ -z "${DRY_RUN:-}" && "${SPEC_ONLY}" == "TRUE" ]] && exit

CMD="spack add ${LBANN_SPEC}"
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

# Explicitly add the lbann spec to the environment
CMD="spack develop --no-clone -p ${LBANN_HOME} ${LBANN_SPEC}"
echo ${CMD} | tee -a ${LOG}
[[ -z "${DRY_RUN:-}" ]] && { ${CMD} || exit_on_failure "${CMD}"; }

CMD="spack install --only dependencies ${INSTALL_DEV_BUILD_EXTRAS} ${LBANN_SPEC}"
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
if [[ ! -d "${BUILD_DIR}" ]]; then
    CMD="mkdir -p ${BUILD_DIR}"
    echo ${CMD}
    [[ -z "${DRY_RUN:-}" ]] && { ${CMD} || exit_on_failure "${CMD}"; }
fi

# Check to see if the link to the build directory exists and is valid
SPACK_BUILD_DIR="spack-build-${LBANN_SPEC_HASH}"
if [[ -L "${SPACK_BUILD_DIR}" ]]; then
  # If the link is not valid or are told to clean it, remove the link
  if [[ ! -d "${SPACK_BUILD_DIR}" || ! -z "${CLEAN_BUILD}" ]]; then
      CMD="rm ${SPACK_BUILD_DIR}"
      echo ${CMD}
      [[ -z "${DRY_RUN:-}" ]] && { ${CMD} || exit_on_failure "${CMD}"; }
  fi
fi

# If the spack build directory does not exist, create a tmp directory and link it
if [[ ! -e "${SPACK_BUILD_DIR}" && -z "${NO_TMP_BUILD_DIR}" && -z "${DRY_RUN:-}" ]]; then
    tmp_dir=$(mktemp -d -t lbann-spack-build-${LBANN_SPEC_HASH}-$(date +%Y-%m-%d-%H%M%S)-XXXXXXXXXX)
    echo ${tmp_dir}
    CMD="ln -s ${tmp_dir} spack-build-${LBANN_SPEC_HASH}"
    echo ${CMD}
    [[ -z "${DRY_RUN:-}" ]] && { ${CMD} || exit_on_failure "${CMD}"; }
fi

##########################################################################################
# Actually install LBANN from local source
# Really you could use the install command, but the dev-build has nice options and better output
CMD="spack dev-build --source-path ${LBANN_HOME} ${DEV_BUILD_FLAGS} ${INSTALL_DEV_BUILD_EXTRAS} ${LBANN_SPEC}"
echo ${CMD} | tee -a ${LOG}
[[ -z "${DRY_RUN:-}" ]] && { ${CMD} || exit_on_failure "${CMD}"; }

# Don't use the output of this file since it will not exist if the compilation is not successful
# LBANN_BUILD_DIR=$(grep "PROJECT_BINARY_DIR:" ${LBANN_HOME}/spack-build-out.txt | awk '{print $2}')

if [[ -L "${LINK_DIR}" ]]; then
    CMD="rm ${LINK_DIR}"
    echo ${CMD} | tee -a ${LOG}
    [[ -z "${DRY_RUN:-}" ]] && { ${CMD} || exit_on_failure "${CMD}"; }
fi

CMD="ln -s ${LBANN_HOME}/spack-build-${LBANN_SPEC_HASH} ${LINK_DIR}"
echo ${CMD} | tee -a ${LOG}
[[ -z "${DRY_RUN:-}" ]] && { ${CMD} || exit_on_failure "${CMD}"; }

##########################################################################################
# Once LBANN is installed deactivate the environment and try to find the package to get the
# installed path
# This is no longer necessary since we don't need to find the compiler version right now
# spack env deactivate
# LBANN_FIND_CMD="spack find --path lbann@${LBANN_LABEL} arch=${SPACK_ARCH} /${LBANN_SPEC_HASH}"
# echo ${LBANN_FIND_CMD} | tee -a ${LOG}
# COMPILER_VER="<FIND ONE>"
# if [[ -z "${DRY_RUN:-}" ]]; then
#     LBANN_PATH=$(${LBANN_FIND_CMD})
#     LBANN_INSTALL_DIR=${LBANN_PATH##* }
#     COMPILER_VER=$(basename $(dirname $LBANN_INSTALL_DIR))
# fi
echo "##########################################################################################" | tee -a ${LOG}
echo "LBANN is installed in a spack environment named ${LBANN_ENV}, access it via:" | tee -a ${LOG}
echo "  spack env activate -p ${LBANN_ENV}" | tee -a ${LOG}
echo "To rebuild LBANN from source drop into a shell with the spack build environment setup:" | tee -a ${LOG}
echo "  spack build-env ${LBANN_SPEC} -- bash" | tee -a ${LOG}
echo "  cd spack-build-${LBANN_SPEC_HASH}" | tee -a ${LOG}
echo "  ninja install" | tee -a ${LOG}
echo "To use this version of LBANN use the module system without the need for activating the environment (does not require being in an environment)" | tee -a ${LOG}
echo "  module load lbann/${LBANN_LABEL}-${LBANN_SPEC_HASH}" | tee -a ${LOG}
echo "or have spack load the module auto-magically. It is installed in a spack environment named ${LBANN_ENV}, access it via: (has to be executed from the environment)"  | tee -a ${LOG}
echo "  spack load lbann@${LBANN_LABEL} arch=${SPACK_ARCH}" | tee -a ${LOG}
echo "##########################################################################################" | tee -a ${LOG}
echo "Alternatively, for rebuilding, the script can drop create a shell in the build environment" | tee -a ${LOG}
echo "  ${BASH_SOURCE} --build-env-only bash -l ${LBANN_LABEL_PREFIX:-local} -- ${CMD_LINE_VARIANTS}" | tee -a ${LOG}
echo "  cd spack-build-${LBANN_SPEC_HASH}" | tee -a ${LOG}
echo "  ninja install" | tee -a ${LOG}
echo "##########################################################################################" | tee -a ${LOG}
echo "All details of the run are logged to ${LOG}"
echo "##########################################################################################"

# Lastly, Save the log file in the build directory
CMD="cp ${LOG} ${LBANN_HOME}/spack-build-${LBANN_SPEC_HASH}/${LOG}"
echo ${CMD}
[[ -z "${DRY_RUN:-}" ]] && ${CMD}

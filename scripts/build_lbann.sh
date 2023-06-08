#!/bin/bash

# Make sure that lmod is setup correctly
if [ -n "${LMOD_PKG}" ]; then
    source $LMOD_PKG/init/bash
fi

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
REUSE_ENV=
DRY_RUN=
CLEAN_BUILD=
ALLOW_BACKEND_BUILDS=
# Flag for passing subcommands to spack dev-build
DEV_BUILD_FLAGS=
# Flag for passing subcommands to spack install
if [[ ${SYS} = "Darwin" ]]; then
BUILD_JOBS="-j $(($(sysctl -n hw.physicalcpu)/2+2))"
else
BUILD_JOBS="-j $(($(nproc)/2+2))"
fi
SPACK_INSTALL_DEPENDENCIES_ONLY=

CONFIG_FILE_NAME=

LBANN_VARIANTS=
CMD_LINE_VARIANTS=

# Default versions of Hydrogen, DiHydrogen, and Aluminum - use head of repo
HYDROGEN_VER="@develop"
ALUMINUM_VER="@1.3.1:"
#ALUMINUM_VER="@1.0.0-lbann"
DIHYDROGEN_VER="@develop"
# Default variants for Conduit to minimize dependencies
CONDUIT_VARIANTS="~hdf5_compat~fortran~parmetis"

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
  ${C}--help${N}                     Display this help message and exit.
  ${C}--ci-pip${N}                   PIP install CI required Python packages
  ${C}--clean-build${N}              Delete the local link to the build directory
  ${C}--clean-deps${N}               Forcibly uninstall Hydrogen, Aluminum, and DiHydrogen dependencies
  ${C}--configure-only${N}           Stop after adding all packages to the environment
  ${C}-c | --config-file${N}         Ingest a CMake config file
  ${C}-d | --define-env${N}          Define (create) a Spack environment, including the lbann dependencies, for building LBANN from local source.  (This wil overwrite any existing environment of the same name)
  ${C}--dependencies-only${N}        Only install the dependencies of the top-level packages (e.g. LBANN)
  ${C}--dry-run${N}                  Dry run the commands (no effect)
  ${C}-e | --extras <PATH>${N}       Add other packages from file at PATH to the Spack environment in addition to LBANN (Flag can be repeated)
  ${C}-j | --build-jobs <N>${N}      Number of parallel processes to use for compiling, e.g. -j \$((\$(nproc)+2))
  ${C}-l | --label <LABEL>${N}       LBANN version label prefix: (default label is local-<SPACK_ARCH_TARGET>,
                             and is built and installed in the spack environment lbann-<label>-<SPACK_ARCH_TARGET>
  ${C}-m | --mirror <PATH>${N}       Specify a Spack mirror (and buildcache)
  ${C}--center-mirrors <PATH>${N}    Use Center-specify mirrors (and buildcache)
  ${C}--no-default-mirrors>${N}      Disable the default set of mirrors
  ${C}--no-modules${N}               Don't try to load any modules (use the existing users environment)
  ${C}-p | --pkg <PACKAGE>${N}       Add package PACKAGE to the Spack environment in addition to LBANN (Flag can be repeated)
  ${C}--pip <requirements.txt>${N}   PIP install Python packages in requirements.txt with the version of Python used by LBANN (Flag can be repeated)
  ${C}-r | --reuse-env${N}           Reuse a Spack environment, including the lbann dependencies, for building LBANN from local source.  (This will create a new one if none is found)
  ${C}--tmp-build-dir${N}            Put the build directory in tmp space
  ${C}--spec-only${N}                Stop after a spack spec command
  ${C}-s | --stable${N}              Use the latest stable defaults not the head of Hydrogen, DiHydrogen and Aluminum repos
  ${C}--hydrogen-repo <PATH>${N}     Use a local repository for the Hydrogen library
  ${C}--dihydrogen-repo <PATH>${N}   Use a local repository for the DiHydrogen library
  ${C}--aluminum-repo <PATH>${N}     Use a local repository for the Aluminum library
  ${C}-u | --user <VERSION>${N}      Build from the GitHub repo -- as a "user" not developer using optional <VERSION> tag
  ${C}--allow-backend-builds${N}     Allow for builds that are not compatible with the host target architecture
  ${C}--${N}                         Pass all variants to spack after the dash dash (--)
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
        --ci-pip)
            PIP_EXTRAS="${PIP_EXTRAS} ${LBANN_HOME}/ci_test/requirements.txt"
            ;;
        --clean-build)
            CLEAN_BUILD="TRUE"
            ;;
        --clean-deps)
            CLEAN_DEPS="TRUE"
            ;;
        --configure-only)
            CONFIGURE_ONLY="TRUE"
            ;;
        -c|--config-file)
            if [ -n "${2}" ]; then
                CONFIG_FILE_NAME=${2}
                shift
            else
                echo "\"${1}\" option requires a non-empty option argument" >&2
                exit 1
            fi
            ;;
        -d|--define-env)
            INSTALL_DEPS="TRUE"
            ;;
        --dependencies-only)
            SPACK_INSTALL_DEPENDENCIES_ONLY="TRUE"
            ;;
        --dry-run)
            DRY_RUN="TRUE"
            ;;
        -e|--extras)
            if [ -n "${2}" ]; then
                EXTRAS="${EXTRAS} ${2}"
                shift
            else
                echo "\"${1}\" option requires a non-empty option argument" >&2
                exit 1
            fi
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
        -j|--build-jobs)
            if [ -n "${2}" ]; then
                BUILD_JOBS="-j${2}"
                shift
            else
                echo "\"${1}\" option requires a non-empty option argument" >&2
                exit 1
            fi
            ;;
        -m|--mirror)
            if [ -n "${2}" ]; then
                USER_MIRROR="${2}"
                shift
            else
                echo "\"${1}\" option requires a non-empty option argument" >&2
                exit 1
            fi
            ;;
        --center-mirrors)
            USE_CENTER_MIRRORS="TRUE"
            ;;
        --no-default-mirrors)
            SKIP_MIRRORS="TRUE"
            ;;
        --no-modules)
            SKIP_MODULES="TRUE"
            ;;
        -p|--pkg)
            if [ -n "${2}" ]; then
                PKG_LIST="${PKG_LIST} ${2}"
                shift
            else
                echo "\"${1}\" option requires a non-empty option argument" >&2
                exit 1
            fi
            ;;
        --pip)
            if [ -n "${2}" ]; then
                PIP_EXTRAS="${PIP_EXTRAS} ${2}"
                shift
            else
                echo "\"${1}\" option requires a non-empty option argument" >&2
                exit 1
            fi
            ;;
        -r|--reuse-env)
            REUSE_ENV="TRUE"
            ;;
        --tmp-build-dir)
            TMP_BUILD_DIR="TRUE"
            ;;
        --spec-only)
            SPEC_ONLY="TRUE"
            ;;
        -s|--stable-defaults)
            # Use the latest released version
            HYDROGEN_VER=
            ALUMINUM_VER="@1.0.0-lbann"
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
        -u|--user)
            USER_BUILD="TRUE"
            if [ -n "${2}" ] && [ ${2:0:1} != "-" ]; then
                LBANN_USER_LABEL=${2}
                shift
            else
                # Use the latest released version
                LBANN_USER_LABEL=
                HYDROGEN_VER=
                ALUMINUM_VER=
                DIHYDROGEN_VER=
            fi
            ;;
        --allow-backend-builds)
            ALLOW_BACKEND_BUILDS="TRUE"
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

# This should be a commit hash (NOT a tag) that needs to exist in the
# spack repository that is checked out. It's a minimum version, so
# more commits is fine.
MIN_SPACK_COMMIT=eb29889f6eae9c415895f34023b84f1d72d01ef4
# New commit should be fa9fb60df332a430ed10ce007d3a07ec90aefcb5

# "spack" is just a shell function; it may not be exported to this
# scope. Just to be sure, reload the shell integration.
if [ -n "${SPACK_ROOT}" ]; then
    pushd ${SPACK_ROOT}
    if git merge-base --is-ancestor ${MIN_SPACK_COMMIT} HEAD &> /dev/null;
    then
        source ${SPACK_ROOT}/share/spack/setup-env.sh
    else
        echo "ERROR: Spack needs at least commit ${MIN_SPACK_COMMIT}."
        HEAD_SHA=$(git rev-parse --verify HEAD)
        echo "ERROR: It is currently at  ${HEAD_SHA}."
        echo "ERROR: Please update spack."
        exit 1
    fi
    popd
else
    echo "Spack required.  Please set SPACK_ROOT environment variable"
    exit 1
fi

SPACK_VERSION=$(spack --version | sed 's/-.*//g' | sed 's/[(].*[)]//g')
MIN_SPACK_VERSION=0.19.1

compare_versions ${SPACK_VERSION} ${MIN_SPACK_VERSION}
VALID_SPACK=$?

if [[ ${VALID_SPACK} -eq 2 ]]; then
    echo "Newer version of Spack required.  Detected version ${SPACK_VERSION} requires at least ${MIN_SPACK_VERSION}"
    exit 1
fi

##########################################################################################
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
SPACK_ARCH_OS=$(spack arch -o)
SPACK_ARCH_GENERIC_TARGET=$(spack python -c "import archspec.cpu as cpu; print(str(cpu.host().family))")
# Create a modified spack arch with generic target architecture
SPACK_ARCH_PLATFORM_GENERIC_TARGET="${SPACK_ARCH_PLATFORM}-${SPACK_ARCH_GENERIC_TARGET}"

if [[ -n "${USER_BUILD:-}" ]]; then
    LBANN_LABEL="${LBANN_USER_LABEL}"
    LBANN_ENV="${LBANN_ENV:-lbann-${LBANN_LABEL_PREFIX:-user}-${SPACK_ARCH_TARGET}}"
else
    LBANN_LABEL="${LBANN_LABEL_PREFIX:-local}-${SPACK_ARCH_TARGET}"
    LBANN_ENV="${LBANN_ENV:-lbann-${LBANN_LABEL}}"
fi
# If a label is defined create a variable with a leading @ symbol
if [[ -n "${LBANN_LABEL:-}" ]]; then
    AT_LBANN_LABEL="@${LBANN_LABEL}"
else
    AT_LBANN_LABEL=""
fi

LOG="spack-build-${LBANN_ENV}.log"
if [[ -f ${LOG} ]]; then
    CMD="rm ${LOG}"
    echo ${CMD}
    [[ -z "${DRY_RUN:-}" ]] && ${CMD}
fi

LBANN_BUILD_LABEL="lbann_${CLUSTER}_${LBANN_LABEL}"
LBANN_BUILD_PARENT_DIR="${LBANN_HOME}/builds/${LBANN_BUILD_LABEL}"
#LBANN_BUILD_PARENT_DIR="${PWD}/builds/${LBANN_BUILD_LABEL}"
LBANN_BUILD_DIR="${LBANN_BUILD_PARENT_DIR}/build"
LBANN_INSTALL_DIR="${LBANN_BUILD_PARENT_DIR}/install"
LBANN_MODFILES_DIR="${LBANN_INSTALL_DIR}/etc/modulefiles"
LBANN_SETUP_FILE="${LBANN_BUILD_PARENT_DIR}/LBANN_${CLUSTER}_${LBANN_LABEL}_setup_build_tools.sh"
LBANN_INSTALL_FILE_LABEL="LBANN_${CLUSTER}_${LBANN_LABEL}_setup_module_path.sh"
LBANN_INSTALL_FILE="${LBANN_HOME}/${LBANN_INSTALL_FILE_LABEL}"
#LBANN_INSTALL_FILE="${PWD}/${LBANN_INSTALL_FILE_LABEL}"

if [[ ! -d "${LBANN_BUILD_PARENT_DIR}" ]]; then
    CMD="mkdir -p ${LBANN_BUILD_PARENT_DIR}"
    echo ${CMD}
    [[ -z "${DRY_RUN:-}" ]] && { ${CMD} || exit_on_failure "${CMD}"; }
fi
##########################################################################################

function exit_on_failure()
{
    local cmd="$1"
    echo -e "FAILED: ${cmd}"
    echo "##########################################################################################" | tee -a ${LOG}
    echo "LBANN is being installed in ${LBANN_INSTALL_DIR}, but an error occured." | tee -a ${LOG}
    echo "To rebuild LBANN go to ${LBANN_BUILD_DIR}, and rerun:" | tee -a ${LOG}
    echo "  cd ${LBANN_BUILD_DIR}" | tee -a ${LOG}
    echo "  ninja install" | tee -a ${LOG}
    echo "If the error occured in the dependencies, they are being installed in a spack environment named ${LBANN_ENV}, access it via:" | tee -a ${LOG}
    echo "  spack env activate -p ${LBANN_ENV}" | tee -a ${LOG}
    echo "  spack install --only dependencies" | tee -a ${LOG}
    echo "  spack install -u initconfig lbann" | tee -a ${LOG}
    echo "##########################################################################################" | tee -a ${LOG}
    echo "All details of the run are logged to ${LOG}"
    echo "##########################################################################################"
    exit 1
}

function exit_with_instructions()
{
    echo "##########################################################################################" | tee -a ${LOG}
    echo "LBANN is installed in ${LBANN_INSTALL_DIR}, access it via:" | tee -a ${LOG}
    echo "  ml use ${LBANN_MODFILES_DIR}" | tee -a ${LOG}
    echo "  ml load lbann" | tee -a ${LOG}
    echo "  lbann_pfe.sh <cmd>" | tee -a ${LOG}
    echo "To rebuild LBANN go to ${LBANN_BUILD_DIR}, and rerun:" | tee -a ${LOG}
    echo "  ${NINJA} install" | tee -a ${LOG}
    echo "To manipulate the dependencies you can activate the spack environment named ${LBANN_ENV} via:" | tee -a ${LOG}
    echo "  spack env activate -p ${LBANN_ENV}" | tee -a ${LOG}
    echo "##########################################################################################" | tee -a ${LOG}
    echo "All details of the run are logged to ${LOG}"
    echo "##########################################################################################"
    exit 1
}

function warn_on_failure()
{
    local cmd="$1"
    echo -e "WARNING CMD FAILED: ${cmd}"
}

##########################################################################################
# Figure out if there are default dependencies or flags (e.g.  MPI/BLAS library) for the center
CENTER_COMPILER_PATHS=
CENTER_COMPILER=
DEPENDENTS_CENTER_COMPILER=
CENTER_DEPENDENCIES=
CENTER_LINKER_FLAGS=
CENTER_BLAS_LIBRARY=
CENTER_PIP_PACKAGES=
CENTER_UPSTREAM_PATH=
set_center_specific_spack_dependencies ${CENTER} ${SPACK_ARCH_TARGET}

if [[ ! "${LBANN_VARIANTS}" =~ .*"^hydrogen".* ]]; then
    # If the user didn't supply a specific version of Hydrogen on the command line add one
    HYDROGEN="^hydrogen${HYDROGEN_VER} ${CENTER_BLAS_LIBRARY}"
fi

if [[ (! "${LBANN_VARIANTS}" =~ .*"^aluminum".*) && (! "${LBANN_VARIANTS}" =~ .*"~al".*) && (! "${CENTER_DEPENDENCIES}" =~ .*"^aluminum".*) ]]; then
    # If the user didn't supply a specific version of Aluminum on the command line add one
    ALUMINUM="^aluminum${ALUMINUM_VER}"
fi

if [[ ! "${LBANN_VARIANTS}" =~ .*"^dihydrogen".* ]]; then
    # If the user didn't supply a specific version of DiHydrogen on the command line add one
    # Due to concretizer errors force the openmp variant for DiHydrogen
    DIHYDROGEN="^dihydrogen${DIHYDROGEN_VER} ${CENTER_BLAS_LIBRARY}"
fi

if [[ ! "${LBANN_VARIANTS}" =~ .*"^conduit".* ]]; then
    # If the user didn't supply a specific set of variants for Condiuit on the command line add one
    CONDUIT="^conduit${CONDUIT_VARIANTS}"
fi

if [[ "${LBANN_VARIANTS}" =~ (.*)(%[0-9a-zA-Z:\.@]+)(.*) ]]; then
    # If the user specified a compiler on the command line, extract it here for use when propagating to
    # other packages
    CENTER_COMPILER=${BASH_REMATCH[2]}
    LBANN_VARIANTS="${BASH_REMATCH[1]} ${BASH_REMATCH[3]}"
fi

if [[ "${CENTER_COMPILER}" =~ .*"%clang".* ]]; then
    # If the compiler is clang use the LLD fast linker
    CENTER_LINKER_FLAGS="+lld"
fi

if [[ "${CENTER_COMPILER}" =~ .*"%gcc".* ]]; then
    # If the compiler is gcc use the gold fast linker
    CENTER_LINKER_FLAGS="+gold"
fi

if [[ -n "${CENTER_PIP_PACKAGES:-}" ]]; then
    PIP_EXTRAS="${PIP_EXTRAS} ${CENTER_PIP_PACKAGES}"
fi

GPU_VARIANTS_ARRAY=('+cuda' '+rocm')
DEPENDENT_PACKAGES_GPU_VARIANTS=
POSSIBLE_AWS_OFI_PLUGIN=
POSSIBLE_DNN_LIB=
POSSIBLE_NVSHMEM_LIB=
for GPU_VARIANTS in ${GPU_VARIANTS_ARRAY[@]}
do
    if [[ "${LBANN_VARIANTS}" =~ .*"${GPU_VARIANTS}".* ]]; then
        # Define the GPU_ARCH_VARIANTS field
        GPU_ARCH_VARIANTS=
        set_center_specific_gpu_arch ${CENTER} ${SPACK_ARCH_TARGET}
        # Prepend the GPU_ARCH_VARIANTS for the LBANN variants if the +cuda variant is defined
        LBANN_VARIANTS=" ${GPU_ARCH_VARIANTS} ${LBANN_VARIANTS}"
        if [[ "${GPU_VARIANTS}" == "+rocm" ]]; then
            # For now, don't forward the amdgpu_target field to downstream packages
            # Py-Torch does not support it
            DEPENDENT_PACKAGES_GPU_VARIANTS="${GPU_VARIANTS}"
            POSSIBLE_AWS_OFI_PLUGIN="aws-ofi-rccl"
        else
            DEPENDENT_PACKAGES_GPU_VARIANTS="${GPU_VARIANTS} ${GPU_ARCH_VARIANTS}"
            POSSIBLE_AWS_OFI_PLUGIN="aws-ofi-nccl"
            POSSIBLE_DNN_LIB="cudnn"
            POSSIBLE_NVSHMEM_LIB="nvshmem"
        fi
    fi
done

# There is a known problem on LC systems with older default compilers and their
# associated C++ std libraries, python, and LBANN.  In these instances force
# spack to build all of them with a consistent set of compilers
# Check if the user explicitly doesn't want Python support inside of LBANN
# or if the center uses standard PIP installed packages then assume that they have
# the right C++ std libraries
if [[ ! "${LBANN_VARIANTS}" =~ .*"~python".* ]]; then
    if [[ -z ${CENTER_PIP_PACKAGES:-} ]]; then
        # If Python support is not disabled add NumPy as an external for sanity
        # Specifically, for use within the data reader, NumPy has to have the same
        # C++ std library
        if [[ ! "${PKG_LIST}" =~ .*"py-numpy".* ]]; then
            PKG_LIST="${PKG_LIST} py-numpy@1.16.0:"
        fi
        # Include PyTest as a top level dependency because of a spack bug that fails
        # to add it for building things like NumPy
        if [[ ! "${PKG_LIST}" =~ .*"py-pytest".* ]]; then
            PKG_LIST="${PKG_LIST} py-pytest"
        fi
    fi
fi

# Record the original command in the log file
echo "${ORIG_CMD}" | tee -a ${LOG}

if [[ ! -n "${SKIP_MODULES:-}" ]]; then
    # Activate modules
    MODULE_CMD=
    set_center_specific_modules ${CENTER} ${SPACK_ARCH_TARGET}
    if [[ "${CENTER_COMPILER}" =~ .*"%clang".* && -n "${MODULE_CMD_CLANG}" && -z "${MODULE_CMD}" ]]; then
        # If the compiler is clang use the specificed set of modules
        MODULE_CMD=${MODULE_CMD_CLANG}
    fi

    if [[ "${CENTER_COMPILER}" =~ .*"%gcc".* && -n "${MODULE_CMD_GCC}" && -z "${MODULE_CMD}" ]]; then
        # If the compiler is gcc use the specificed set of modules
        MODULE_CMD=${MODULE_CMD_GCC}
    fi

    if [[ -n ${MODULE_CMD} ]]; then
        echo ${MODULE_CMD} | tee -a ${LOG}
        [[ -z "${DRY_RUN:-}" ]] && { eval ${MODULE_CMD} || exit_on_failure "${MODULE_CMD}"; }
    fi
fi

# If there is a request to reuse the "environment" look for a config file too
#
if [[ -n "${REUSE_ENV:-}" || -z "${INSTALL_DEPS:-}" ]]; then
    if [[ -n "${CONFIG_FILE_NAME}" ]]; then
        echo "Both the reuse flag (-r) and a config file flag (-c) were provide, favor the config file: ${CONFIG_FILE_NAME}"
        if [[ ! -e "${CONFIG_FILE_NAME}" || ! -r "${CONFIG_FILE_NAME}" ]]; then
            echo "Unable to find or read ${CONFIG_FILE_NAME}"
            exit 1
        fi
    else
        find_cmake_config_file ${LBANN_LABEL} ${CENTER_COMPILER} ${LBANN_HOME}
        if [[ ! -z "${MATCHED_CONFIG_FILE_PATH}" ]]; then
            if [[ -e "${MATCHED_CONFIG_FILE_PATH}" && -r "${MATCHED_CONFIG_FILE_PATH}" ]]; then
                echo "I have found and will use ${MATCHED_CONFIG_FILE_PATH}"
                CONFIG_FILE_NAME=${MATCHED_CONFIG_FILE}
                if [[ ! -e "${LBANN_BUILD_PARENT_DIR}/${CONFIG_FILE_NAME}" ]]; then
                    echo "Overwritting exising CMake config file in ${LBANN_BUILD_PARENT_DIR}/${CONFIG_FILE_NAME}"
                fi
                # Save the config file in the build directory
                CMD="cp ${MATCHED_CONFIG_FILE_PATH} ${LBANN_BUILD_PARENT_DIR}/${CONFIG_FILE_NAME}"
                echo ${CMD}
                [[ -z "${DRY_RUN:-}" ]] && { ${CMD} || warn_on_failure "${CMD}"; }
            fi
        fi
    fi
fi

# If a config file is provided skip everything
if [[ -z "${CONFIG_FILE_NAME}" ]]; then

# If the user asks to resuse an environment see if it exists, if not set one up
if [[ -n "${REUSE_ENV:-}" ]]; then
    # Check to make sure that both the -d and -r flags are not concurrently set
    if [[ -n "${INSTALL_DEPS:-}" ]]; then
        [[ -z "${DRY_RUN:-}" ]] && { exit_on_failure "Invalid combination of -r and -d flags"; }
    fi
    # Look for existing environment with the same name
    if [[ $(spack env list | grep -e "${LBANN_ENV}$") ]]; then
        echo "Spack environment ${LBANN_ENV} already exists... reusing it"
    else
        echo "Spack environment ${LBANN_ENV} does not exists... creating it (as if -d flag was thrown)"
        INSTALL_DEPS="TRUE"
    fi
fi

##########################################################################################
# Set an upstream spack repository that is holding standard dependencies
if [[ -r "${CENTER_UPSTREAM_PATH:-}" ]]; then
    EXISTING_UPSTREAM=`spack config get upstreams`
    if [[ ${EXISTING_UPSTREAM} == "upstreams: {}" ]]; then
        read -p "Do you want to add pointer for this spack repository to ${CENTER_UPSTREAM_PATH} (y/N): " response
        if [[ ${response^^} == "Y" ]]; then
            CMD="spack config --scope site add upstreams:spack-lbann-vast:install_tree:${CENTER_UPSTREAM_PATH}"
            echo ${CMD} | tee -a ${LOG}
            [[ -z "${DRY_RUN:-}" ]] && { ${CMD} || exit_on_failure "${CMD}"; }
        fi
    else
        printf "Spack is using\n${EXISTING_UPSTREAM}\n"
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

if [[ -z "${USER_BUILD:-}" ]]; then
    ##########################################################################################
    # For developer builds uninstall any existing versions for this architecture with the same label
    # -- note that this has to be done outside of an environment
    # For finding the lbann version don't use the architecture because sometimes it is "downgraded"
    LBANN_FIND_CMD="spack find --format {hash:7} lbann${AT_LBANN_LABEL}"
    echo ${LBANN_FIND_CMD} | tee -a ${LOG}
    LBANN_HASH=$(${LBANN_FIND_CMD})
    if [[ -n "${LBANN_HASH}" && ! "${LBANN_HASH}" =~ "No package matches the query" ]]; then
        LBANN_HASH_ARRAY=(${LBANN_HASH})
        for h in ${LBANN_HASH_ARRAY[@]}
        do
            CMD="spack uninstall -y --force lbann${AT_LBANN_LABEL} /${h}"
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

##########################################################################################
# Activate the environment
CMD="spack env activate -p ${LBANN_ENV}"
echo ${CMD} | tee -a ${LOG}
if [[ -z "${DRY_RUN:-}" ]]; then
    if [[ -z $(spack env list | grep -e "${LBANN_ENV}$") ]]; then
        echo "Spack could not activate environment ${LBANN_ENV} -- install dependencies with -d flag"
        exit 1
    fi
    ${CMD} || exit_on_failure "${CMD}"
fi

##########################################################################################
# Force a unified environment
if [[ -n "${INSTALL_DEPS:-}" ]]; then
    # Force the environment to concretize together with any additional packages
    CMD="spack config add concretizer:unify:true"
    echo ${CMD} | tee -a ${LOG}
    [[ -z "${DRY_RUN:-}" ]] && { ${CMD} || exit_on_failure "${CMD}"; }
fi

##########################################################################################
# See if the is a local spack mirror or buildcache
if [[ -n "${USER_MIRROR:-}" ]]; then
    # Allow the user to overwrite a standard mirror
    MIRRORS="${MIRRORS:-} ${USER_MIRROR}"
fi

if [[ -n "${INSTALL_DEPS:-}" && -z "${SKIP_MIRRORS:-}" ]]; then
    # https://cache.spack.io/tag/develop/
    CMD="spack mirror add spack-build-cache-develop https://binaries.spack.io/develop"
    echo ${CMD} | tee -a ${LOG}
    [[ -z "${DRY_RUN:-}" ]] && { ${CMD} || exit_on_failure "${CMD}"; }
    # Tell Spack to trust the keys in the build cache
    CMD="spack buildcache keys --install --trust"
    echo ${CMD} | tee -a ${LOG}
    [[ -z "${DRY_RUN:-}" ]] && { ${CMD} || exit_on_failure "${CMD}"; }
fi

if [[ -n "${INSTALL_DEPS:-}" && -n "${MIRRORS:-}" ]]; then
    i=0
    for MIRROR in ${MIRRORS}
    do
        if [[ -r "${MIRROR:-}" ]]; then
            CMD="spack mirror add lbann${i} ${MIRROR}"
            echo ${CMD} | tee -a ${LOG}
            [[ -z "${DRY_RUN:-}" ]] && { ${CMD} || exit_on_failure "${CMD}"; }
            i=$((${i}+1))

            # Tell Spack to trust the keys in the build cache
            CMD="spack buildcache keys --install --trust"
            echo ${CMD} | tee -a ${LOG}
            [[ -z "${DRY_RUN:-}" ]] && { ${CMD} || exit_on_failure "${CMD}"; }

            # Manually force Spack to trust the keys in the build cache - this is a hack until
            # https://github.com/spack/spack/issues/23186 is fixed
            if [[ -e "${MIRROR}/build_cache/_pgp/B180FE4A5ECF4C02D21E6A67F13D1FBB0E55F96F.pub" ]]; then
                CMD="spack gpg trust ${MIRROR}/build_cache/_pgp/B180FE4A5ECF4C02D21E6A67F13D1FBB0E55F96F.pub"
                echo ${CMD} | tee -a ${LOG}
                [[ -z "${DRY_RUN:-}" ]] && { ${CMD} || exit_on_failure "${CMD}"; }
            fi
        fi
    done
fi

##########################################################################################
# Establish the spec for LBANN
LBANN_SPEC="lbann${AT_LBANN_LABEL} ${CENTER_COMPILER} ${CENTER_LINKER_FLAGS} ${LBANN_VARIANTS} ${HYDROGEN} ${DIHYDROGEN} ${ALUMINUM} ${CONDUIT} ${CENTER_DEPENDENCIES}"
##########################################################################################

##########################################################################################
# Add things to the environment
##########################################################################################
SPACK_SOLVE_EXTRA_PACKAGES=
if [[ -n "${INSTALL_DEPS:-}" ]]; then
    # Set the environment to use CURL rather than url fetcher since it has issues
    # on LC platforms
    CMD="spack config add config:url_fetch_method:curl"
    echo ${CMD} | tee -a ${LOG}
    [[ -z "${DRY_RUN:-}" ]] && { ${CMD} || exit_on_failure "${CMD}"; }

    # Set the environment to avoid concretizing for microarchitectures that are
    # incompatible with the current host on LC platforms
    if [[ -z "${ALLOW_BACKEND_BUILDS:-}" ]]; then
        CMD="spack config add concretizer:targets:host_compatible:true"
        echo ${CMD} | tee -a ${LOG}
        [[ -z "${DRY_RUN:-}" ]] && { ${CMD} || exit_on_failure "${CMD}"; }
    fi

    # See if there are any center-specific externals
    SPACK_ENV_YAML_FILE="${SPACK_ROOT}/var/spack/environments/${LBANN_ENV}/spack.yaml"
    CMD="set_center_specific_externals ${CENTER} ${SPACK_ARCH_TARGET} ${SPACK_ARCH} ${SPACK_ENV_YAML_FILE}"
    echo ${CMD} | tee -a ${LOG}
    [[ -z "${DRY_RUN:-}" ]] && { ${CMD} || exit_on_failure "${CMD}"; }

    if [[ -n "${DEPENDENT_PACKAGES_GPU_VARIANTS:-}" ]]; then
        # Force the environment to concretize with the same set of GPU variants
        CMD="spack config add packages:all:variants:'${DEPENDENT_PACKAGES_GPU_VARIANTS}'"
        echo ${CMD} | tee -a ${LOG}
        [[ -z "${DRY_RUN:-}" ]] && { `spack config add packages:all:variants:"${DEPENDENT_PACKAGES_GPU_VARIANTS}"` || exit_on_failure "${CMD}"; }
    fi

    # Put the compilers into the SITE scope so that we can execute
    # spack load commands later without activating the environment
    CMD="spack compiler find --scope site ${CENTER_COMPILER_PATHS}"
    echo ${CMD} | tee -a ${LOG}
    [[ -z "${DRY_RUN:-}" ]] && { ${CMD} || exit_on_failure "${CMD}"; }

    # Limit the scope of the external search to minimize overhead time
    # CRAY_MANIFEST="/opt/cray/pe/cpe-descriptive-manifest"
    # if [[ -e ${CRAY_MANIFEST} ]]; then
    #    CMD="spack external read-cray-manifest --directory ${CRAY_MANIFEST} --fail-on-error"
    #    echo ${CMD} | tee -a ${LOG}
    #    [[ -z "${DRY_RUN:-}" ]] && { ${CMD} || exit_on_failure "${CMD}"; }
    # fi

    # Use standard tags for common packages
    CMD="spack external find --scope env:${LBANN_ENV} --tag core-packages --tag build-tools --tag rocm"
    echo ${CMD} | tee -a ${LOG}
    [[ -z "${DRY_RUN:-}" ]] && { ${CMD} || exit_on_failure "${CMD}"; }

    CMD="spack external find --scope env:${LBANN_ENV} bzip2 cuda cudnn hwloc libfabric nccl ncurses openblas perl python rccl rdma-core sqlite spectrum-mpi mvapich2 openmpi"
    echo ${CMD} | tee -a ${LOG}
    [[ -z "${DRY_RUN:-}" ]] && { ${CMD} || exit_on_failure "${CMD}"; }

    CMD="cleanup_clang_compilers ${CENTER} ${SPACK_ARCH_OS} ${SPACK_ENV_YAML_FILE}"
    echo ${CMD} | tee -a ${LOG}
    [[ -z "${DRY_RUN:-}" ]] && { ${CMD} || exit_on_failure "${CMD}"; }

    ##########################################################################################
    # Tell the spack environment to use a local repository for these libraries
    if [[ -n "${HYDROGEN_PATH:-}" ]]; then
        CMD="spack develop --no-clone -p ${HYDROGEN_PATH} hydrogen${HYDROGEN_VER}"
        echo "${CMD}" | tee -a ${LOG}
        [[ -z "${DRY_RUN:-}" ]] && { ${CMD} || exit_on_failure "${CMD}"; }
    fi

    if [[ -n "${DIHYDROGEN_PATH:-}" ]]; then
        CMD="spack develop --no-clone -p ${DIHYDROGEN_PATH} dihydrogen${DIHYDROGEN_VER}"
        echo "${CMD}" | tee -a ${LOG}
        [[ -z "${DRY_RUN:-}" ]] && { ${CMD} || exit_on_failure "${CMD}"; }
    fi

    if [[ -n "${ALUMINUM_PATH:-}" ]]; then
        CMD="spack develop --no-clone -p ${ALUMINUM_PATH} aluminum${ALUMINUM_VER}"
        echo "${CMD}" | tee -a ${LOG}
        [[ -z "${DRY_RUN:-}" ]] && { ${CMD} || exit_on_failure "${CMD}"; }
    fi
    ##########################################################################################

    # Explicitly add the lbann spec to the environment
    CMD="spack add ${LBANN_SPEC}"
    echo ${CMD} | tee -a ${LOG}
    [[ -z "${DRY_RUN:-}" ]] && { ${CMD} || exit_on_failure "${CMD}"; }

    # Explicitly mark lbann for development
    if [[ -z "${USER_BUILD:-}" ]]; then
        # Only "develop" the lbann package with the version number not the entire
        # spec, because the spec is already handled with the add command.  Including the
        # entire spec in the develop command triggers a bug in Spack v0.17.1 where the
        # environment cannot be built twich with the --reuse flag
        CMD="spack develop --no-clone -p ${LBANN_HOME} lbann${AT_LBANN_LABEL}"
        echo ${CMD} | tee -a ${LOG}
        [[ -z "${DRY_RUN:-}" ]] && { ${CMD} || exit_on_failure "${CMD}"; }
    fi

    # Add any extra packages in file EXTRAS that you want to build in conjuction with the LBANN package
    if [[ -n "${EXTRAS:-}" ]]; then
        for e in ${EXTRAS}
        do
            CMD="source ${e}"
            echo ${CMD} | tee -a ${LOG}
            [[ -z "${DRY_RUN:-}" ]] && { ${CMD} || exit_on_failure "${CMD}"; }
        done
    fi

    # Add any extra packages specified on the command line that you want to build in conjuction with the LBANN package
    if [[ -n "${PKG_LIST:-}" ]]; then
        if [[ -z ${DEPENDENTS_CENTER_COMPILER} ]]; then
            DEPENDENTS_CENTER_COMPILER=${CENTER_COMPILER}
        fi
        for p in ${PKG_LIST}
        do
            CMD="spack add ${p} ${DEPENDENTS_CENTER_COMPILER}"
            SPACK_SOLVE_EXTRA_PACKAGES="${p} ${DEPENDENTS_CENTER_COMPILER} ${SPACK_SOLVE_EXTRA_PACKAGES}"
            echo ${CMD} | tee -a ${LOG}
            [[ -z "${DRY_RUN:-}" ]] && { ${CMD} || exit_on_failure "${CMD}"; }
        done
    fi
fi # [[ -n "${INSTALL_DEPS:-}" ]]

CMD="spack solve -l ${LBANN_SPEC} ${SPACK_SOLVE_EXTRA_PACKAGES}"
if [[ "${SPEC_ONLY}" == "TRUE" ]]; then
   echo ${CMD} | tee -a ${LOG}
   if [[ -z "${DRY_RUN:-}" ]]; then
       eval ${CMD} || exit_on_failure "${CMD}\nIf the error is that boostrapping failed try something like 'module load gcc/8.3.1; spack compiler add' and then rerunning"
   fi
fi

if [[ -n "${INSTALL_DEPS:-}" ]]; then
  # Try to concretize the environment and catch the return code
  CMD="spack concretize"
  echo ${CMD} | tee -a ${LOG}
  [[ -z "${DRY_RUN:-}" ]] && { ${CMD} || exit_on_failure "${CMD}"; }
fi

# Get the spack hash for LBANN (Ensure that the concretize command has been run so that any impact of external packages is factored in)
LBANN_SPEC_HASH=$(spack find -cl | grep -v "\-\-\-\-\-\-" | grep lbann${AT_LBANN_LABEL} | awk '{print $1}')

# Get the spack hash for aws-ofi plugin (Ensure that the concretize command has been run so that any impact of external packages is factored in)
if [[ -n "${POSSIBLE_AWS_OFI_PLUGIN}" ]]; then
    AWS_OFI_PLUGIN_SPEC_HASH=$(spack find -cl | grep -v "\-\-\-\-\-\-" | grep "${POSSIBLE_AWS_OFI_PLUGIN}@" | awk '{print $1}')
    if [[ -n "${AWS_OFI_PLUGIN_SPEC_HASH}" ]]; then
        echo "LBANN built with AWS plugin ${AWS_OFI_PLUGIN_SPEC_HASH} for ${POSSIBLE_AWS_OFI_PLUGIN}"
    fi
fi

if [[ -n "${POSSIBLE_DNN_LIB}" ]]; then
    DNN_LIB_SPEC_HASH=$(spack find -cl | grep -v "\-\-\-\-\-\-" | grep "${POSSIBLE_DNN_LIB}@" | awk '{print $1}')
    if [[ -n "${DNN_LIB_SPEC_HASH}" ]]; then
        echo "LBANN built with DNN library ${DNN_LIB_SPEC_HASH} for ${POSSIBLE_DNN_LIB}"
    fi
fi

if [[ -n "${POSSIBLE_NVSHMEM_LIB}" ]]; then
    NVSHMEM_LIB_SPEC_HASH=$(spack find -cl | grep -v "\-\-\-\-\-\-" | grep "${POSSIBLE_NVSHMEM_LIB}@" | awk '{print $1}')
    if [[ -n "${NVSHMEM_LIB_SPEC_HASH}" ]]; then
        echo "LBANN built with NVSHMEM library ${NVSHMEM_LIB_SPEC_HASH} for ${POSSIBLE_NVSHMEM_LIB}"
    fi
fi

# If SPEC_ONLY was requested bail
[[ -z "${DRY_RUN:-}" && "${SPEC_ONLY}" == "TRUE" ]] && exit_with_instructions

# If the user only wants to configure the environment
[[ ${CONFIGURE_ONLY:-} ]] && exit_with_instructions

##########################################################################################
# Actually install LBANN's dependencies from local source
CMD="spack install --only dependencies ${BUILD_JOBS}"
echo ${CMD} | tee -a ${LOG}
[[ -z "${DRY_RUN:-}" ]] && { ${CMD} || exit_on_failure "${CMD}"; }

if [[ -n "${SPACK_INSTALL_DEPENDENCIES_ONLY:-}" ]]; then
    echo "Finished installing dependencies.  Exiting..."
    exit
fi


##########################################################################################
# Install any other packages to make sure that PYTHONPATH is properly setup
# Install any other top level packages requested
if [[ -n "${PKG_LIST:-}" ]]; then
    for p in ${PKG_LIST}
    do
        CMD="spack install ${BUILD_JOBS} ${p}"
        echo ${CMD} | tee -a ${LOG}
        [[ -z "${DRY_RUN:-}" ]] && { ${CMD} || exit_on_failure "${CMD}"; }
    done
fi

# Install any extra Python packages via PIP if requested
if [[ -n "${PIP_EXTRAS:-}" ]]; then
    for p in ${PIP_EXTRAS}
    do
        CMD="python3 -m pip install -r ${p}"
        echo ${CMD} | tee -a ${LOG}
        [[ -z "${DRY_RUN:-}" ]] && { ${CMD} || exit_on_failure "${CMD}"; }
    done
fi

##########################################################################################
# Configure but don't install LBANN using spack
CMD="spack install -u initconfig ${BUILD_JOBS} lbann"
echo ${CMD} | tee -a ${LOG}
[[ -z "${DRY_RUN:-}" ]] && { ${CMD} || exit_on_failure "${CMD}"; }

# SPECIFIC_COMPILER=$(spack find --format "{prefix} {version} {name},/{hash} {compiler}" conduit | cut -f4 -d" ")

if [[ ! -d "${LBANN_BUILD_DIR}" ]]; then
    CMD="mkdir -p ${LBANN_BUILD_DIR}"
    echo ${CMD}
    [[ -z "${DRY_RUN:-}" ]] && { ${CMD} || exit_on_failure "${CMD}"; }
fi

if [[ -z "${DRY_RUN:-}" ]]; then
    # Record which cmake was used to build this
    LBANN_CMAKE=$(spack build-env lbann -- which cmake)
    # Record which ninja was used to build this
    LBANN_NINJA=$(spack build-env lbann -- which ninja)
    # Record which python was used to build this
    LBANN_PYTHON=$(spack build-env lbann -- which python3)
    LBANN_PYTHONPATH=$(spack build-env lbann -- printenv PYTHONPATH)

cat > ${LBANN_SETUP_FILE}<<EOF
export LBANN_CMAKE=${LBANN_CMAKE}
export LBANN_NINJA=${LBANN_NINJA}
export LBANN_PYTHON=${LBANN_PYTHON}
export LBANN_PYTHONPATH=${LBANN_PYTHONPATH}
export LBANN_CMAKE_DIR=\$(dirname ${LBANN_CMAKE})
export LBANN_NINJA_DIR=\$(dirname ${LBANN_NINJA})
export LBANN_PYTHON_DIR=\$(dirname ${LBANN_PYTHON})
# Postpend the paths to the build tools to avoid putting system paths up front
export PATH=\${PATH}:\${LBANN_CMAKE_DIR}:\${LBANN_NINJA_DIR}:\${LBANN_PYTHON_DIR}
export PYTHONPATH=\${LBANN_PYTHONPATH}:\${PYTHONPATH}
EOF

CMD="chmod +x ${LBANN_SETUP_FILE}"
echo ${CMD} | tee -a ${LOG}
[[ -z "${DRY_RUN:-}" ]] && { ${CMD} || warn_on_failure "${CMD}"; }

cat > ${LBANN_INSTALL_FILE}<<EOF
# Directory structure used for this build
export LBANN_BUILD_LABEL=${LBANN_BUILD_LABEL}
export LBANN_BUILD_PARENT_DIR=${LBANN_BUILD_PARENT_DIR}
export LBANN_BUILD_DIR=${LBANN_BUILD_DIR}
export LBANN_INSTALL_DIR=${LBANN_INSTALL_DIR}
export LBANN_MODFILES_DIR=${LBANN_MODFILES_DIR}
export LBANN_SETUP_FILE=${LBANN_SETUP_FILE}
EOF

ENV_ROOT_PKG_LIST=$(spack find -x --format "{name}")
if [[ -n "${ENV_ROOT_PKG_LIST:-}" ]]; then
    for p in ${ENV_ROOT_PKG_LIST}
    do
        PKG_PYTHONPATH=$(spack build-env ${p} -- printenv PYTHONPATH)
        if [[ -n "${PKG_PYTHONPATH}" ]]; then
            P_ENV=$(echo "${p}" | tr '-' '_')
cat >> ${LBANN_INSTALL_FILE}<<EOF
# Add PYTHONPATH for top level python package: ${p}
export ${P_ENV}_PKG_PYTHONPATH=${PKG_PYTHONPATH}
export PYTHONPATH=\${${P_ENV}_PKG_PYTHONPATH}:\${PYTHONPATH}
EOF
        fi
    done
fi

if [[ -n "${MODULE_CMD}" ]]; then
cat >> ${LBANN_INSTALL_FILE}<<EOF
# Modules loaded during this installation
${MODULE_CMD}
EOF
cat >> ${LBANN_SETUP_FILE}<<EOF
# Modules loaded during this installation
${MODULE_CMD}
EOF
fi

if [[ -n "${AWS_OFI_PLUGIN_SPEC_HASH}" ]]; then
cat >> ${LBANN_INSTALL_FILE}<<EOF
# Key spack pacakges that have to be loaded at runtime to ensure behavior outside of
# a spack environment matches behavior inside of a runtime environment
spack load ${POSSIBLE_AWS_OFI_PLUGIN} /${AWS_OFI_PLUGIN_SPEC_HASH}
EOF
fi

if [[ -n "${DNN_LIB_SPEC_HASH}" ]]; then
cat >> ${LBANN_INSTALL_FILE}<<EOF
# Key spack pacakges that have to be loaded at runtime to ensure behavior outside of
# a spack environment matches behavior inside of a runtime environment
spack load ${POSSIBLE_DNN_LIB} /${DNN_LIB_SPEC_HASH}
EOF
fi

if [[ -n "${NVSHMEM_LIB_SPEC_HASH}" ]]; then
cat >> ${LBANN_INSTALL_FILE}<<EOF
# Key spack pacakges that have to be loaded at runtime to ensure behavior outside of
# a spack environment matches behavior inside of a runtime environment
spack load ${POSSIBLE_NVSHMEM_LIB} /${NVSHMEM_LIB_SPEC_HASH}
EOF
fi

# Setup the module use path last in case the modules cmd purges the system
cat >> ${LBANN_INSTALL_FILE}<<EOF
echo "BVE about to update the model path"
printenv MODULEPATH
ml use ${LBANN_MODFILES_DIR}
export MODULEPATH=\${MODULEPATH}
echo "BVE just updated the model path"
printenv MODULEPATH
EOF

CMD="chmod +x ${LBANN_INSTALL_FILE}"
echo ${CMD} | tee -a ${LOG}
[[ -z "${DRY_RUN:-}" ]] && { ${CMD} || warn_on_failure "${CMD}"; }
fi

# Save the install file in the build directory
if [[ ! -e ${LBANN_BUILD_PARENT_DIR}/${LBANN_INSTALL_FILE_LABEL} ]]; then
    echo "Overwritting exising install file in ${LBANN_BUILD_PARENT_DIR}/${LBANN_INSTALL_FILE_LABEL}"
fi
CMD="cp ${LBANN_INSTALL_FILE} ${LBANN_BUILD_PARENT_DIR}/${LBANN_INSTALL_FILE_LABEL}"
echo ${CMD} | tee -a ${LOG}
[[ -z "${DRY_RUN:-}" ]] && { ${CMD} || warn_on_failure "${CMD}"; }

# Drop out of the environment for the rest of the build
CMD="spack env deactivate"
echo ${CMD} | tee -a ${LOG}
[[ -z "${DRY_RUN:-}" ]] && { ${CMD} || exit_on_failure "${CMD}"; }

# Now that the config file is generated set the field
find_cmake_config_file ${LBANN_LABEL} ${CENTER_COMPILER} ${LBANN_HOME}
if [[ ! -z "${MATCHED_CONFIG_FILE_PATH}" ]]; then
    if [[ -e "${MATCHED_CONFIG_FILE_PATH}" && -r "${MATCHED_CONFIG_FILE_PATH}" ]]; then
        echo "I have found and will use ${MATCHED_CONFIG_FILE}"
        CONFIG_FILE_NAME=${MATCHED_CONFIG_FILE}
        if [[ ! -e "${LBANN_BUILD_PARENT_DIR}/${CONFIG_FILE_NAME}" ]]; then
            echo "Overwritting exising CMake config file in ${LBANN_BUILD_PARENT_DIR}/${CONFIG_FILE_NAME}"
        fi
        # Save the config file in the build directory
        CMD="cp ${MATCHED_CONFIG_FILE_PATH} ${LBANN_BUILD_PARENT_DIR}/${CONFIG_FILE_NAME}"
        echo ${CMD} | tee -a ${LOG}
        [[ -z "${DRY_RUN:-}" ]] && { ${CMD} || warn_on_failure "${CMD}"; }
    else
        echo "ERROR: Unable to open the generated config file: ${MATCHED_CONFIG_FILE_PATH}"
        exit 1
    fi
else
    echo "ERROR: Unable to find the generated config file for: ${LBANN_LABEL} ${CENTER_COMPILER} in ${LBANN_HOME}"
    exit 1
fi

fi # [[ ! -z "${CONFIG_FILE_NAME}" ]]

# Check to see if the link to the build directory exists and is valid
#SPACK_BUILD_DIR="spack-build-${LBANN_SPEC_HASH}"
if [[ -L "${LBANN_BUILD_DIR}" ]]; then
  # If the link is not valid or are told to clean it, remove the link
  if [[ ! -d "${LBANN_BUILD_DIR}" || ! -z "${CLEAN_BUILD}" ]]; then
      CMD="rm ${LBANN_BUILD_DIR}"
      echo ${CMD} | tee -a ${LOG}
      [[ -z "${DRY_RUN:-}" ]] && { ${CMD} || exit_on_failure "${CMD}"; }
  fi
fi

# If there is a directory there and we are told to clean it, remove the directory
if [[ -d "${LBANN_BUILD_DIR}" && ! -z "${CLEAN_BUILD}" ]]; then
    CMD="rm -r ${LBANN_BUILD_DIR}"
    echo ${CMD} | tee -a ${LOG}
    [[ -z "${DRY_RUN:-}" ]] && { ${CMD} || exit_on_failure "${CMD}"; }
fi

# If the spack build directory does not exist, create a directory or tmp directory and link it
if [[ ! -d "${LBANN_BUILD_DIR}" ]]; then
    if [[ -n "${TMP_BUILD_DIR:-}" && -z "${DRY_RUN:-}" ]]; then
        tmp_dir=$(mktemp -d -t ${LBANN_BUILD_LABEL}-$(date +%Y-%m-%d-%H%M%S)-XXXXXXXXXX)
        echo ${tmp_dir}
        CMD="ln -s ${tmp_dir} ${LBANN_BUILD_DIR}"
        echo ${CMD} | tee -a ${LOG}
        [[ -z "${DRY_RUN:-}" ]] && { ${CMD} || exit_on_failure "${CMD}"; }
    else
        CMD="mkdir -p ${LBANN_BUILD_DIR}"
        echo ${CMD} | tee -a ${LOG}
        [[ -z "${DRY_RUN:-}" ]] && { ${CMD} || exit_on_failure "${CMD}"; }
    fi
fi

if [[ -e "${LBANN_SETUP_FILE}" && -r "${LBANN_SETUP_FILE}" ]]; then
    echo "I have found and will use ${LBANN_SETUP_FILE}"
    source ${LBANN_SETUP_FILE}
else
    echo "ERROR: Unable to find the setup build tools file: ${LBANN_SETUP_FILE}"
    echo "ERROR: Please reinstall the dependencies (-d) to recreate the file."
    exit 1
fi

CMAKE_CMD="${LBANN_CMAKE} -C ${LBANN_HOME}/${CONFIG_FILE_NAME} -B ${LBANN_BUILD_DIR} -DCMAKE_INSTALL_PREFIX=${LBANN_INSTALL_DIR} ${LBANN_HOME}"
echo ${CMAKE_CMD} | tee -a ${LOG}
[[ -z "${DRY_RUN:-}" ]] && { ${CMAKE_CMD} || exit_on_failure "${CMAKE_CMD}"; }

CMD="cd ${LBANN_BUILD_DIR}"
echo ${CMD} | tee -a ${LOG}
[[ -z "${DRY_RUN:-}" ]] && { ${CMD} || exit_on_failure "${CMD}"; }

CMD="${LBANN_NINJA} install"
echo ${CMD} | tee -a ${LOG}
[[ -z "${DRY_RUN:-}" ]] && { ${CMD} || exit_on_failure "${CMD}"; }

CMD="ml use ${LBANN_MODFILES_DIR}"
echo ${CMD} | tee -a ${LOG}
[[ -z "${DRY_RUN:-}" ]] && { ${CMD} || exit_on_failure "${CMD}"; }

# Don't use the output of this file since it will not exist if the compilation is not successful
# LBANN_BUILD_DIR=$(grep "PROJECT_BINARY_DIR:" ${LBANN_HOME}/spack-build-out.txt | awk '{print $2}')

if [[ -z "${USER_BUILD:-}" ]]; then
    # Copy the compile_commands.json file to LBANN_HOME
    if [[ -e "${LBANN_HOME}/spack-build-${LBANN_SPEC_HASH}/compile_commands.json" ]]; then
        CMD="cp ${LBANN_HOME}/spack-build-${LBANN_SPEC_HASH}/compile_commands.json ${LBANN_HOME}/compile_commands.json"
        echo ${CMD} | tee -a ${LOG}
        [[ -z "${DRY_RUN:-}" ]] && { ${CMD} || exit_on_failure "${CMD}"; }
    fi
fi

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
echo "LBANN is installed in ${LBANN_INSTALL_DIR}, access it via:" | tee -a ${LOG}
echo "  source ${LBANN_INSTALL_FILE}" | tee -a ${LOG}
echo "  ml use ${LBANN_MODFILES_DIR}" | tee -a ${LOG}
echo "  ml load lbann" | tee -a ${LOG}
echo "  lbann_pfe.sh <cmd>" | tee -a ${LOG}
echo "To rebuild LBANN go to ${LBANN_BUILD_DIR}, and rerun:" | tee -a ${LOG}
if [[ -n ${MODULE_CMD} ]]; then
    echo "  ${MODULE_CMD}" | tee -a ${LOG}
fi
echo "  source ${LBANN_SETUP_FILE}" | tee -a ${LOG}
echo "  ${CMAKE_CMD}" | tee -a ${LOG}
echo "  ${LBANN_NINJA} install" | tee -a ${LOG}
echo "To manipulate the dependencies you can activate the spack environment named ${LBANN_ENV} via:" | tee -a ${LOG}
echo "  spack env activate -p ${LBANN_ENV}" | tee -a ${LOG}
echo "To manipulate the version of python used it is:" | tee -a ${LOG}
echo "  ${LBANN_PYTHON}" | tee -a ${LOG}
# if [[ -z "${USER_BUILD:-}" ]]; then
#     echo "To rebuild LBANN from source drop into a shell with the spack build environment setup (requires active environment):" | tee -a ${LOG}
#     echo "  spack build-env lbann -- bash" | tee -a ${LOG}
#     echo "  cd spack-build-${LBANN_SPEC_HASH}" | tee -a ${LOG}
#     echo "  ninja install" | tee -a ${LOG}
# fi
echo "Additional Python packages for working with LBANN can be added either via PIP or by concretizing them together in spack., activate the spack environment then" | tee -a ${LOG}
echo "To install them via PIP: 1) the spack environment (see above) and 2) issue the following command" | tee -a ${LOG}
echo "  python3 -m pip install -r <requirements file>" | tee -a ${LOG}
echo "To install them via Spack: include them on the build_lbann.sh script command line argument via -e <path to text file of packages> or -p <spack package name>" | tee -a ${LOG}
echo "##########################################################################################" | tee -a ${LOG}
echo "All details of the run are logged to ${LOG}"
echo "##########################################################################################"

if [[ -z "${USER_BUILD:-}" ]]; then
    if [[ ! -e "${LBANN_BUILD_DIR}/${LOG}" ]]; then
        # Lastly, Save the log file in the build directory
        CMD="cp ${LOG} ${LBANN_BUILD_DIR}/${LOG}"
        echo ${CMD} | tee -a ${LOG}
        [[ -z "${DRY_RUN:-}" ]] && { ${CMD} || warn_on_failure "${CMD}"; }
    fi
fi

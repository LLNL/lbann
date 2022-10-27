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

LBANN_VARIANTS=
CMD_LINE_VARIANTS=

# Default versions of Hydrogen, DiHydrogen, and Aluminum - use head of repo
HYDROGEN_VER="@develop"
ALUMINUM_VER="@1.0.0-lbann"
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
  ${C}-d | --define-env${N}          Define (create) a Spack environment, including the lbann dependencies, for building LBANN from local source.  (This wil overwrite any existing environment of the same name)
  ${C}--dry-run${N}                  Dry run the commands (no effect)
  ${C}-e | --extras <PATH>${N}       Add other packages from file at PATH to the Spack environment in addition to LBANN (Flag can be repeated)
  ${C}-j | --build-jobs <N>${N}      Number of parallel processes to use for compiling, e.g. -j \$((\$(nproc)+2))
  ${C}-l | --label <LABEL>${N}       LBANN version label prefix: (default label is local-<SPACK_ARCH_TARGET>,
                             and is built and installed in the spack environment lbann-<label>-<SPACK_ARCH_TARGET>
  ${C}-m | --mirror <PATH>${N}       Specify a Spack mirror (and buildcache)
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
  ${C}--update-buildcache <PATH>${N} Update a buildcache defined by the Spack mirror (Expert Only)
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
        -d|--define-env)
            INSTALL_DEPS="TRUE"
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
            CLEAN_BUILD="TRUE"
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
        --update-buildcache)
            if [ -n "${2}" ] && [ ${2:0:1} != "-" ]; then
                UPDATE_BUILDCACHE=${2}
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
MIN_SPACK_COMMIT=47bfc60845b71830ee54a04c597419c7eedd2a42

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
MIN_SPACK_VERSION=0.18.0

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

function exit_on_failure()
{
    local cmd="$1"
    echo -e "FAILED: ${cmd}"
    echo "##########################################################################################" | tee -a ${LOG}
    echo "LBANN is being installed in a spack environment named ${LBANN_ENV} but an error occured, access it via:" | tee -a ${LOG}
    echo "  spack env activate -p ${LBANN_ENV}" | tee -a ${LOG}
    echo "To rebuild LBANN from source drop into a shell with the spack build environment setup (requires active environment):" | tee -a ${LOG}
    echo "  spack build-env lbann -- bash" | tee -a ${LOG}
    echo "  cd spack-build-${LBANN_SPEC_HASH}" | tee -a ${LOG}
    echo "  ninja install" | tee -a ${LOG}
    echo "##########################################################################################" | tee -a ${LOG}
    echo "All details of the run are logged to ${LOG}"
    echo "##########################################################################################"
    exit 1
}

function exit_with_instructions()
{
    echo "##########################################################################################" | tee -a ${LOG}
    echo "LBANN is being installed in a spack environment named ${LBANN_ENV}, access it via:" | tee -a ${LOG}
    echo "  spack env activate -p ${LBANN_ENV}" | tee -a ${LOG}
    echo "To finish installing LBANN and its dependencies (requires active environment):" | tee -a ${LOG}
    echo "  spack install" | tee -a ${LOG}
    echo "Once the initial installation is complete, to rebuild LBANN from source drop into a shell with the spack build environment setup (requires active environment):" | tee -a ${LOG}
    echo "  spack build-env lbann -- bash" | tee -a ${LOG}
    echo "  cd spack-build-${LBANN_SPEC_HASH}" | tee -a ${LOG}
    echo "  ninja install" | tee -a ${LOG}
    echo "Once installed, to use this version of LBANN use the module system without the need for activating the environment (does not require being in an environment)" | tee -a ${LOG}
    echo "  module load lbann/${LBANN_LABEL}-${LBANN_SPEC_HASH}" | tee -a ${LOG}
    echo "or have spack load the module auto-magically. It is installed in a spack environment named ${LBANN_ENV}, access it via: (has to be executed from the environment)"  | tee -a ${LOG}
    echo "  spack load lbann${AT_LBANN_LABEL} arch=${SPACK_ARCH}" | tee -a ${LOG}
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
CENTER_COMPILER=
CENTER_DEPENDENCIES=
CENTER_LINKER_FLAGS=
CENTER_BLAS_LIBRARY=
CENTER_PIP_PACKAGES=
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
        else
            DEPENDENT_PACKAGES_GPU_VARIANTS="${GPU_VARIANTS} ${GPU_ARCH_VARIANTS}"
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
    CMD="spack mirror add binary_mirror  https://binaries.spack.io/releases/v0.18"
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

    CMD="spack compiler find --scope env:${LBANN_ENV}"
    echo ${CMD} | tee -a ${LOG}
    [[ -z "${DRY_RUN:-}" ]] && { ${CMD} || exit_on_failure "${CMD}"; }

    # Limit the scope of the external search to minimize overhead time
    CRAY_MANIFEST="/opt/cray/pe/cpe-descriptive-manifest"
    if [[ -e ${CRAY_MANIFEST} ]]; then
       CMD="spack external read-cray-manifest --directory ${CRAY_MANIFEST} --fail-on-error"
       echo ${CMD} | tee -a ${LOG}
       [[ -z "${DRY_RUN:-}" ]] && { ${CMD} || exit_on_failure "${CMD}"; }
    fi

    # Use standard tags for common packages
    CMD="spack external find --scope env:${LBANN_ENV} --tag core-packages --tag build-tools --tag rocm"
    echo ${CMD} | tee -a ${LOG}
    [[ -z "${DRY_RUN:-}" ]] && { ${CMD} || exit_on_failure "${CMD}"; }

    CMD="spack external find --scope env:${LBANN_ENV} bzip2 cuda cudnn hipblas hwloc nccl ncurses perl python rccl rdma-core sqlite spectrum-mpi mvapich2 openmpi"
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

    ##########################################################################################
    # If this build is going to go to a buildcache, pad out the install tree so that it can be relocated
    # Don't mix this with normal installtions, duplicate packages can get installed
    if [[ -n "${UPDATE_BUILDCACHE:-}" ]]; then
        spack config add "config:install_tree:padded_length:128"
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
        for p in ${PKG_LIST}
        do
            CMD="spack add ${p} ${CENTER_COMPILER}"
            SPACK_SOLVE_EXTRA_PACKAGES="${p} ${CENTER_COMPILER} ${SPACK_SOLVE_EXTRA_PACKAGES}"
            echo ${CMD} | tee -a ${LOG}
            [[ -z "${DRY_RUN:-}" ]] && { ${CMD} || exit_on_failure "${CMD}"; }
        done
    fi
fi

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
# If SPEC_ONLY was requested bail
[[ -z "${DRY_RUN:-}" && "${SPEC_ONLY}" == "TRUE" ]] && exit_with_instructions

# If the user only wants to configure the environment
[[ ${CONFIGURE_ONLY:-} ]] && exit_with_instructions

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

# If there is a directory there and we are told to clean it, remove the directory
if [[ -d "${SPACK_BUILD_DIR}" && ! -z "${CLEAN_BUILD}" ]]; then
    CMD="rm -r ${SPACK_BUILD_DIR}"
    echo ${CMD}
    [[ -z "${DRY_RUN:-}" ]] && { ${CMD} || exit_on_failure "${CMD}"; }
fi

# If the spack build directory does not exist, create a tmp directory and link it
if [[ ! -e "${SPACK_BUILD_DIR}" && -n "${TMP_BUILD_DIR:-}" && -z "${DRY_RUN:-}" ]]; then
    tmp_dir=$(mktemp -d -t lbann-spack-build-${LBANN_SPEC_HASH}-$(date +%Y-%m-%d-%H%M%S)-XXXXXXXXXX)
    echo ${tmp_dir}
    CMD="ln -s ${tmp_dir} spack-build-${LBANN_SPEC_HASH}"
    echo ${CMD}
    [[ -z "${DRY_RUN:-}" ]] && { ${CMD} || exit_on_failure "${CMD}"; }
fi

##########################################################################################
# Actually install LBANN from local source
CMD="spack install ${BUILD_JOBS}"
echo ${CMD} | tee -a ${LOG}
[[ -z "${DRY_RUN:-}" ]] && { ${CMD} || exit_on_failure "${CMD}"; }

if [[ -n "${UPDATE_BUILDCACHE:-}" && -r "${UPDATE_BUILDCACHE:-}" ]]; then
    # Make sure that all of the packages in the environment are in the mirror
    CMD="spack mirror create -d ${UPDATE_BUILDCACHE} --all"
    echo ${CMD} | tee -a ${LOG}
    # Don't check the return code of the mirror create command, it will fail to install some packages
    [[ -z "${DRY_RUN:-}" ]] && ${CMD}

    if [[ ! -e "${UPDATE_BUILDCACHE}/pubring.gpg" ]]; then
        CMD="cp ${SPACK_ROOT}/opt/spack/gpg/pubring.gpg ${UPDATE_BUILDCACHE}/pubring.gpg"
        echo ${CMD} | tee -a ${LOG}
        [[ -z "${DRY_RUN:-}" ]] && { ${CMD} || exit_on_failure "${CMD}"; }
    fi

    SPACK_INSTALL_ROOT=$(grep root $SPACK_ROOT/etc/spack/config.yaml | awk '{ print $2 }')
    for ii in $(spack find --format "{prefix} {version} {name},/{hash}" |
        grep -v -E "^(develop^master)" |
        grep -e "${SPACK_ROOT}" -e "${SPACK_INSTALL_ROOT}" |
        cut -f3 -d" ")
    do
        NAME=${ii%,*};
        HASH=${ii#*,};
        case ${NAME} in
            "cuda" | "cudnn" | "ncurses" | "openssl" | "lbann")
                echo "Skipping $ii"
                continue
                ;;
        esac
        CMD="spack buildcache check --rebuild-on-error --mirror-url file://${UPDATE_BUILDCACHE} -s ${HASH}"
        echo -e "${NAME}:\t ${CMD}" | tee -a ${LOG}
        if [[ -z "${DRY_RUN:-}" ]]; then
            if ${CMD};
            then
                true
            else
                CMD="spack buildcache create -af -d ${UPDATE_BUILDCACHE} --only=package ${HASH}"
                echo ${CMD} | tee -a ${LOG}
                [[ -z "${DRY_RUN:-}" ]] && { ${CMD} || exit_on_failure "${CMD}"; }
            fi
        fi
    done
    CMD="spack buildcache update-index -d ${UPDATE_BUILDCACHE}"
    echo ${CMD} | tee -a ${LOG}
    [[ -z "${DRY_RUN:-}" ]] && { ${CMD} || exit_on_failure "${CMD}"; }
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
echo "LBANN is installed in a spack environment named ${LBANN_ENV}, access it via:" | tee -a ${LOG}
echo "  spack env activate -p ${LBANN_ENV}" | tee -a ${LOG}
if [[ -z "${USER_BUILD:-}" ]]; then
    echo "To rebuild LBANN from source drop into a shell with the spack build environment setup (requires active environment):" | tee -a ${LOG}
    echo "  spack build-env lbann -- bash" | tee -a ${LOG}
    echo "  cd spack-build-${LBANN_SPEC_HASH}" | tee -a ${LOG}
    echo "  ninja install" | tee -a ${LOG}
fi
echo "Additional Python packages for working with LBANN can be added either via PIP or by concretizing them together in spack., activate the spack environment then" | tee -a ${LOG}
echo "To install them via PIP: 1) the spack environment (see above) and 2) issue the following command" | tee -a ${LOG}
echo "  python3 -m pip install -r <requirements file>" | tee -a ${LOG}
echo "To install them via Spack: include them on the build_lbann.sh script command line argument via -e <path to text file of packages> or -p <spack package name>" | tee -a ${LOG}
echo "##########################################################################################" | tee -a ${LOG}
echo "All details of the run are logged to ${LOG}"
echo "##########################################################################################"

if [[ -z "${USER_BUILD:-}" ]]; then
    # Lastly, Save the log file in the build directory
    CMD="cp ${LOG} ${LBANN_HOME}/spack-build-${LBANN_SPEC_HASH}/${LOG}"
    echo ${CMD}
    [[ -z "${DRY_RUN:-}" ]] && { ${CMD} || warn_on_failure "${CMD}"; }
fi

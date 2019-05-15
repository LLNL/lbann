#!/bin/bash

# Detect system parameters
CLUSTER=$(hostname | sed 's/\([a-zA-Z][a-zA-Z]*\)[0-9]*/\1/g')
TOSS=$(uname -r | sed 's/\([0-9][0-9]*\.*\)\-.*/\1/g')
ARCH=$(uname -m)
CORAL=$([[ $(hostname) =~ (sierra|lassen|ray) ]] && echo 1 || echo 0)

################################################################
# Default options
################################################################

COMPILER=gnu
if [ "${CLUSTER}" == "surface" -o "${CLUSTER}" == "pascal" ]; then
    module load gcc/7.3.0
    module load opt cudatoolkit/9.2
elif [ "${CLUSTER}" == "sierra" -o "${CLUSTER}" == "lassen" ]; then
    module load gcc/7.3.1
fi
if [ "${ARCH}" == "x86_64" ]; then
    MPI=mvapich2
elif [ "${ARCH}" == "ppc64le" ]; then
    MPI=spectrum
fi
BUILD_TYPE=Release
Elemental_DIR=
case $TOSS in
	3.10.0|4.11.0|4.14.0)
		OpenCV_DIR=""
		if [ "${ARCH}" == "x86_64" ]; then
			export VTUNE_DIR=/usr/tce/packages/vtune/default
		elif [ "${ARCH}" == "ppc64le" ]; then
			export VTUNE_DIR=
		fi
		;;
	*)
      OpenCV_DIR=/usr/gapps/brain/tools/OpenCV/2.4.13
      export VTUNE_DIR=/usr/local/tools/vtune
	  ;;
esac
if [ "${ARCH}" == "x86_64" ]; then
    IPPROOT=/p/lscratchh/brainusr/ippicv_lnx
fi


ELEMENTAL_MATH_LIBS=
PATCH_OPENBLAS=ON
C_FLAGS=
CXX_FLAGS=-DLBANN_SET_EL_RNG
Fortran_FLAGS=
CLEAN_BUILD=0
DATATYPE=float
VERBOSE=0
CMAKE_INSTALL_MESSAGE=LAZY
MAKE_NUM_PROCESSES=$(($(nproc) + 1))
NINJA_NUM_PROCESSES=0 # Let ninja decide
GEN_DOC=0
INSTALL_LBANN=0
BUILD_TOOL="make"
BUILD_DIR=
INSTALL_DIR=
BUILD_SUFFIX=
DETERMINISTIC=OFF
WITH_CUDA=
WITH_CUDA_2=ON
WITH_TOPO_AWARE=ON
INSTRUMENT=
WITH_ALUMINUM=
ALUMINUM_WITH_MPI_CUDA=OFF
ALUMINUM_WITH_NCCL=
WITH_CONDUIT=ON
WITH_TBINF=OFF
RECONFIGURE=0
USE_NINJA=0
# In case that autoconf fails during on-demand buid on surface, try the newer
# version of autoconf installed under '/p/lscratchh/brainusr/autoconf/bin'
# by putting it at the beginning of the PATH or use the preinstalled library
# by enabling LIBJPEG_TURBO_DIR
WITH_LIBJPEG_TURBO=ON
#LIBJPEG_TURBO_DIR="/p/lscratchh/brainusr/libjpeg-turbo-1.5.2"

function version_gt() { test "$(printf '%s\n' "$@" | sort -V | head -n 1)" != "$1"; }

if [ "${CLUSTER}" == "surface" ]; then
    AUTOCONF_CUSTOM_DIR=/p/lscratchh/brainusr/autoconf/bin
    AUTOCONF_VER_DEFAULT=`autoconf --version | awk '(FNR==1){print $NF}'`
    AUTOCONF_VER_CUSTOM=`${AUTOCONF_CUSTOM_DIR}/autoconf --version | awk '(FNR==1){print $NF}'`

    if version_gt ${AUTOCONF_VER_CUSTOM} ${AUTOCONF_VER_DEFAULT}; then
        PATH=${AUTOCONF_CUSTOM_DIR}:${PATH}
    fi
fi

################################################################
# Help message
################################################################

function help_message {
    local SCRIPT=$(basename ${0})
    local N=$(tput sgr0)    # Normal text
    local C=$(tput setf 4)  # Colored text
    cat << EOF
Build LBANN on an LLNL LC system.
Can be called anywhere in the LBANN project tree.
Usage: ${SCRIPT} [options]
Options:
  ${C}--help${N}                  Display this help message and exit.
  ${C}--compiler${N} <val>        Specify compiler ('gnu' or 'intel' or 'clang').
  ${C}--mpi${N} <val>             Specify MPI library ('mvapich2' or 'openmpi' or 'spectrum').
  ${C}--datatype${N} <val>        Datatype size in bytes (4 for float and 8 for double).
  ${C}--verbose${N}               Verbose output.
  ${C}--debug${N}                 Build with debug flag.
  ${C}--tbinf${N}                 Build with Tensorboard interface.
  ${C}--vtune${N}                 Build with VTune profiling libraries.
  ${C}--nvprof${N}                Build with region annotations for NVPROF.
  ${C}--reconfigure${N}           Reconfigure build. Used when build parameters are changed (e.g current build is release and --debug is desired). Clean build overrides
  ${C}--clean-build${N}           Clean build directory before building.
  ${C}--make-processes${N} <val>  Number of parallel processes for make.
  ${C}--doc${N}                   Generate documentation.
  ${C}--install-lbann${N}         Install LBANN headers and dynamic library into the build directory.
  ${C}--build${N}                 Specify alternative build directory; default is <lbann_home>/build.
  ${C}--suffix${N}                Specify suffix for build directory. If you are, e.g, building on surface, your build will be <someplace>/surface.llnl.gov, regardless of your choice of compiler or other flags. This option enables you to specify, e.g: --suffix gnu_debug, in which case your build will be in the directory <someplace>/surface.llnl.gov.gnu_debug
  ${C}--instrument${N}            Use -finstrument-functions flag, for profiling stack traces
  ${C}--disable-cuda${N}          Disable CUDA
  ${C}--disable-topo-aware${N}    Disable topological-aware configuration (no HWLOC)
  ${C}--disable-aluminum${N}           Disable the Aluminum communication library
  ${C}--aluminum-with-mpi-cuda         Enable MPI-CUDA backend in Aluminum
  ${C}--disable-aluminum-with-nccl     Disable the NCCL backend in Aluminum
  ${C}--with-conduit              Build with conduit interface
  ${C}--ninja                     Generate ninja files instead of makefiles
  ${C}--ninja-processes${N} <val> Number of parallel processes for ninja.
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
            exit 0
            ;;
        --build)
            # Change default build directory
            if [ -n "${2}" ]; then
                BUILD_DIR=${2}
                shift
            else
                echo "\"${1}\" option requires a non-empty option argument" >&2
                exit 1
            fi
            ;;
        --suffix)
            # Specify suffix for build directory
            if [ -n "${2}" ]; then
                BUILD_SUFFIX=${2}
                shift
            else
                echo "\"${1}\" option requires a non-empty option argument" >&2
                exit 1
            fi
            ;;
        --compiler)
            # Choose compiler
            if [ -n "${2}" ]; then
                COMPILER=${2}
                shift
            else
                echo "\"${1}\" option requires a non-empty option argument" >&2
                exit 1
            fi
            ;;
        --mpi)
            # Choose mpi library
            if [ -n "${2}" ]; then
                MPI=${2}
                shift
            else
                echo "\"${1}\" option requires a non-empty option argument" >&2
                exit 1
            fi
            ;;
        --datatype)
            # Set datatype size
            if [ -n "${2}" ]; then
                DATATYPE=${2}
                shift
            else
                echo "\"${1}\" option requires a non-empty option argument" >&2
                exit 1
            fi
            ;;
        --ninja)
            USE_NINJA=1
            BUILD_TOOL="ninja"
            ;;
        --ninja-processes)
            if [ -n "${2}" ]; then
                NINJA_NUM_PROCESSES=${2}
                shift
            else
                echo "\"${1}\" option requires a non-empty option argument" >&2
                exit 1
            fi
            ;;
        -v|--verbose)
            # Verbose output
            VERBOSE=1
            CMAKE_INSTALL_MESSAGE=ALWAYS
            ;;
        -d|--debug)
            # Debug mode
            BUILD_TYPE=Debug
            DETERMINISTIC=ON
            ;;
        --tbinf)
            # Tensorboard interface
            WITH_TBINF=ON
            ;;
        --vtune)
            # VTune libraries
            WITH_VTUNE=ON
            ;;
        --nvprof)
            WITH_NVPROF=ON
            ;;
        --clean-build|--build-clean)
            # Clean build directory
            CLEAN_BUILD=1
            ;;
        -j|--make-processes)
            if [ -n "${2}" ]; then
                MAKE_NUM_PROCESSES=${2}
                shift
            else
                echo "\"${1}\" option requires a non-empty option argument" >&2
                exit 1
            fi
            ;;
        --doc)
            # Generate documentation
            GEN_DOC=1
            ;;
        -i|--install-lbann)
            INSTALL_LBANN=1
            ;;
        --disable-cuda)
            WITH_CUDA=OFF
            WITH_CUDA_2=OFF
            ;;
        --disable-topo-aware)
            WITH_TOPO_AWARE=OFF
            ;;
        --disable-aluminum)
            WITH_ALUMINUM=OFF
            ;;
        --aluminum-with-mpi-cuda)
            WITH_ALUMINUM=ON
            ALUMINUM_WITH_MPI_CUDA=ON
            ;;
        --disable-aluminum-with-nccl)
            ALUMINUM_WITH_NCCL=OFF
            ;;
        --with-conduit)
            WITH_CONDUIT=ON
            ;;
        --instrument)
            INSTRUMENT="-finstrument-functions -ldl"
            ;;
        --reconfigure)
            RECONFIGURE=1
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

################################################################
# Initialize package system
################################################################

# Determine whether system uses modules
USE_MODULES=0
case $TOSS in
	3.10.0|4.11.0|4.14.0)
		USE_MODULES=1
		;;
	2.6.32)
		USE_MODULES=0
		;;
	*)
		# Initialize modules
		. /usr/share/[mM]odules/init/bash
		USE_MODULES=1
		;;
esac

# Initialize Dotkit if system doesn't use modules
if [ ${USE_MODULES} -eq 0 ]; then
    . /usr/local/tools/dotkit/init.sh
fi

# Load packages
if [ ${USE_MODULES} -ne 0 ]; then
    module load git
    module load cmake/3.12.1
else
    use git
fi

################################################################
# Initialize C/C++/Fortran compilers
################################################################

# Get compiler directory
COMPILER_=${COMPILER}
if [ ${USE_MODULES} -ne 0 ]; then
    if [ "${COMPILER_}" == "gnu" ]; then
        COMPILER_=gcc
    fi
    if [ -z "$(module list 2>&1 | grep ${COMPILER_})" ]; then
        if [ "${COMPILER_}" == "gcc" ]; then
            # Special case to avoid GCC 8.1
            # Note: This should be removed once the bug in GCC 8.1 is
            # patched. See https://github.com/LLNL/lbann/issues/529.
            COMPILER_=$(module --terse spider ${COMPILER_} 2>&1 | grep -v 8.1.0 | sed '/^$/d' | tail -1)
        else
            COMPILER_=$(module --terse spider ${COMPILER_} 2>&1 | sed '/^$/d' | tail -1)
        fi
        module load ${COMPILER_}
    fi
    if [ -z "$(module list 2>&1 | grep ${COMPILER_})" ]; then
        echo "Could not load module (${COMPILER_})"
        exit 1
    fi
    COMPILER_BASE="$(module show ${COMPILER_} 2>&1 | grep '\"PATH\"' | cut -d ',' -f 2 | cut -d ')' -f 1 | sed 's/\/bin//' | sed 's/\"//g')"
else
    if [ "${COMPILER_}" == "intel" ]; then
        COMPILER_=ic-17.0.174
    elif [ "${COMPILER_}" == "gnu" ]; then
        COMPILER_=gcc-4.9.3p
    fi
    if [ -z "$(use | grep ${COMPILER_})" ]; then
        use ${COMPILER_}
    fi
    COMPILER_="$(use | sed 's/ //g' | egrep -e "^${COMPILER_}")"
    if [ -z "${COMPILER_}" ]; then
        echo "Could not load dotkit for ${COMPILER_}"
        exit 1
    fi
    COMPILER_BASE="$(use -hv ${COMPILER_} | grep 'dk_alter PATH' | awk '{print $3}' | sed 's/\/bin//')"
fi
COMPILER_STRIP="$(echo ${COMPILER_} | sed "s/[^[a-z]//g")"
# Get compiler paths
if [ "${COMPILER}" == "gnu" ]; then
    # GNU compilers
    C_COMPILER=${COMPILER_BASE}/bin/gcc
    CXX_COMPILER=${COMPILER_BASE}/bin/g++
    Fortran_COMPILER=${COMPILER_BASE}/bin/gfortran
    COMPILER_VERSION=$(${C_COMPILER} --version | head -n 1 | awk '{print $3}')
    FORTRAN_LIB=${COMPILER_BASE}/lib64/libgfortran.so
elif [ "${COMPILER}" == "intel" ]; then
    # Intel compilers
    C_COMPILER=${COMPILER_BASE}/bin/icc
    CXX_COMPILER=${COMPILER_BASE}/bin/icpc
    Fortran_COMPILER=${COMPILER_BASE}/bin/ifort
    COMPILER_VERSION=$(${C_COMPILER} --version | head -n 1 | awk '{print $3}')
    FORTRAN_LIB=${COMPILER_BASE}/lib/intel64/libifcoremt.so.5
    CXX_FLAGS="${CXX_FLAGS} -std=c++11"
elif [ "${COMPILER}" == "clang" ]; then
    # clang
    # clang depends on gnu fortran library. so, find the dependency
    if [[ ${CORAL} -eq 1 ]]; then
        #gccdep=`ldd ${COMPILER_BASE}/lib/*.so 2> /dev/null | grep gcc | awk '(NF>2){print $3}' | sort | uniq | head -n 1`
        #GCC_VERSION=`ls -l $gccdep | awk '{print $NF}' | cut -d '-' -f 2 | cut -d '/' -f 1`
        # Forcing to gcc 4.9.3 because of the current way of ray's gcc and various clang installation
        GCC_VERSION=4.9.3
        GNU_DIR=/usr/tce/packages/gcc/gcc-${GCC_VERSION}
    elif [ ${USE_MODULES} -ne 0 ]; then
        GCC_VERSION=$(ldd ${COMPILER_BASE}/lib/libclang.so | grep gcc- | awk 'BEGIN{FS="/"} {for (i=1;i<=NF; i++) if ($i~/^gcc-/) print $i}' | tail -n 1)
        GNU_DIR=/usr/tce/packages/gcc/${GCC_VERSION}
    else
        GCC_VERSION=$(ldd ${COMPILER_BASE}/lib/libclang.so | grep gnu | awk 'BEGIN{FS="/"} {for (i=1;i<=NF; i++) if ($i~/^gnu$/) print $(i+1)}' | tail -n 1)
        GNU_DIR=/usr/apps/gnu/${GCC_VERSION}
    fi
    C_COMPILER=${COMPILER_BASE}/bin/clang
    CXX_COMPILER=${COMPILER_BASE}/bin/clang++
    Fortran_COMPILER=${GNU_DIR}/bin/gfortran
    FORTRAN_LIB=${GNU_DIR}/lib64/libgfortran.so
    COMPILER_VERSION=$(${C_COMPILER} --version | awk '(($1=="clang")&&($2=="version")){print $3}')
    MPICH_FC=${GNU_DIR}/bin/gfortran
    #MPI_Fortran_COMPILER="${MPI_DIR}/bin/mpifort -fc=${Fortran_COMPILER}" $ done by exporting MPICH_FC
else
    # Unrecognized compiler
    echo "Unrecognized compiler (${COMPILER})"
    exit 1
fi
# Add compiler optimization flags
if [ "${BUILD_TYPE}" == "Release" ]; then
    if [ "${COMPILER}" == "gnu" ]; then
        C_FLAGS="${C_FLAGS} -O3 ${INSTRUMENT}"
        CXX_FLAGS="${CXX_FLAGS} -O3 ${INSTRUMENT}"
        Fortran_FLAGS="${Fortran_FLAGS} -O3"
        if [ "${CLUSTER}" == "catalyst" ]; then
            C_FLAGS="${C_FLAGS} -march=ivybridge -mtune=ivybridge"
            CXX_FLAGS="${CXX_FLAGS} -march=ivybridge -mtune=ivybridge"
            Fortran_FLAGS="${Fortran_FLAGS} -march=ivybridge -mtune=ivybridge"
        elif [ "${CLUSTER}" == "quartz" ] || [ "${CLUSTER}" == "pascal" ] ; then
            C_FLAGS="${C_FLAGS} -march=broadwell -mtune=broadwell"
            CXX_FLAGS="${CXX_FLAGS} -march=broadwell -mtune=broadwell"
            Fortran_FLAGS="${Fortran_FLAGS} -march=broadwell -mtune=broadwell"
        elif [ "${CLUSTER}" == "surface" ]; then
            C_FLAGS="${C_FLAGS} -march=sandybridge -mtune=sandybridge"
            CXX_FLAGS="${CXX_FLAGS} -march=sandybridge -mtune=sandybridge"
            Fortran_FLAGS="${Fortran_FLAGS} -march=sandybridge -mtune=sandybridge"
        elif [ "${CLUSTER}" == "ray" ]; then
            C_FLAGS="${C_FLAGS} -mcpu=power8 -mtune=power8"
            CXX_FLAGS="${CXX_FLAGS} -mcpu=power8 -mtune=power8"
            Fortran_FLAGS="${Fortran_FLAGS} -mcpu=power8 -mtune=power8"
        elif [ "${CLUSTER}" == "sierra" -o "${CLUSTER}" == "lassen" ]; then
            C_FLAGS="${C_FLAGS} -mcpu=power9 -mtune=power9"
            CXX_FLAGS="${CXX_FLAGS} -mcpu=power9 -mtune=power9"
            Fortran_FLAGS="${Fortran_FLAGS} -mcpu=power9 -mtune=power9"
        fi
    fi
else
    if [ "${COMPILER}" == "gnu" ]; then
        C_FLAGS="${C_FLAGS} -g ${INSTRUMENT}"
        CXX_FLAGS="${CXX_FLAGS} -g ${INSTRUMENT}"
        Fortran_FLAGS="${Fortran_FLAGS} -g"
    fi
fi

# Add flag for libldl: may be needed some compilers
CXX_FLAGS="${CXX_FLAGS} -ldl"
C_FLAGS="${CXX_FLAGS}"


# Set environment variables
CC=${C_COMPILER}
CXX=${CXX_COMPILER}


################################################################
# Initialize directories
################################################################

# Get LBANN root directory
ROOT_DIR=$(realpath $(dirname $0)/..)

# Initialize build directory
if [ -z "${BUILD_DIR}" ]; then
    BUILD_DIR=${ROOT_DIR}/build/${COMPILER}.${BUILD_TYPE}.${CLUSTER}.llnl.gov
fi
if [ -n "${BUILD_SUFFIX}" ]; then
    BUILD_DIR=${BUILD_DIR}.${BUILD_SUFFIX}
fi
mkdir -p ${BUILD_DIR}

# Initialize install directory
if [ -z "${INSTALL_DIR}" ]; then
    INSTALL_DIR=${BUILD_DIR}/install
fi
mkdir -p ${INSTALL_DIR}

SUPERBUILD_DIR="${ROOT_DIR}/superbuild"


################################################################
# Initialize MPI compilers
################################################################

# Invalid MPI libraries
if [ "${ARCH}" == "x86_64" ] && [ "${MPI}" != "mvapich2" ] && [ "${MPI}" != "openmpi" ]; then
    echo "Invalid MPI library selected (${MPI})"
    exit 1
fi
if [ "${ARCH}" == "ppc64le" ] && [ "${MPI}" != "spectrum" ]; then
    echo "Invalid MPI library selected (${MPI})"
    exit 1
fi

if [ "${MPI}" == "spectrum" ]; then
    MPI=spectrum-mpi
fi

if [ -z "${MPI_HOME}" ]; then
	if [ ${USE_MODULES} -ne 0 ]; then
		if [ -z "$(module list 2>&1 | grep ${MPI})" ]; then
			MPI=$(module --terse spider ${MPI} 2>&1 | sed '/^$/d' | tail -1)
			module load ${MPI}
		fi
		if [ -z "$(module list 2>&1 | grep ${MPI})" ]; then
			echo "Could not load module (${MPI})"
			exit 1
		fi
		MPI_HOME=$(module show ${MPI} 2>&1 | grep '\"PATH\"' | cut -d ',' -f 2 | cut -d ')' -f 1 | sed 's/\/bin//' | sed 's/\"//g')
	else
		# The idea here is to check if the module of the specified mpi type is loaded
		MPI_DOTKIT=$(use | grep ${MPI} | sed 's/ //g')
		if [ -z "${MPI_DOTKIT}" ]; then
			if [ "${COMPILER}" == "gnu" ] || [ "${COMPILER}" == "intel" ] || [ "${COMPILER}" == "pgi" ] ; then
				MPI_COMPILER=-${COMPILER}
			elif [ "${COMPILER}" == "clang" ]; then
				MPI_COMPILER=-gnu
			fi
			# The default MVAPICH version does not work on surface
			if [ "${CLUSTER}" == "surface" -a "${MPI}" == "mvapich2" ]; then
				MPI_VERSION="-2.2"
			else
				MPI_VERSION=""
			fi
		else
			MPI_COMPILER=-$(echo ${MPI_DOTKIT} | awk 'BEGIN{FS="-"}{print $2}')
			MPI_VERSION=-$(echo ${MPI_DOTKIT} |  awk 'BEGIN{FS="-"}{print $NF}')
		fi
		if [ "${BUILD_TYPE}" == "Debug" ]; then
			MPI_DEBUG="-debug"
		else
			MPI_DEBUG=""
		fi
		MPI_DOTKIT=${MPI}${MPI_COMPILER}${MPI_DEBUG}${MPI_VERSION}
		echo "Using ${MPI_DOTKIT}"
		use ${MPI_DOTKIT}
		if [ -z "$(use | grep ${MPI_DOTKIT})" ]; then
			echo "Could not load dotkit (${MPI_DOTKIT})"
			exit 1
		fi
		if [ "${COMPILER}" == "gnu" ] || [ "${COMPILER}" == "intel" ] || [ "${COMPILER}" == "pgi" ]; then
			if [ "`echo ${MPI_DOTKIT} | grep ${COMPILER}`" == "" ] ; then
				echo "switch to an MPI version that is consistent with (${COMPILER}) compilers"
				exit 1
			fi
		fi
		MPI_HOME=$(use -hv ${MPI_DOTKIT} | grep 'dk_alter PATH' | awk '{print $3}' | sed 's/\/bin//')
	fi
fi

# Get MPI compilers
export MPI_HOME
export CMAKE_PREFIX_PATH=${MPI_HOME}:${CMAKE_PREFIX_PATH}
export MPI_C_COMPILER=${MPI_HOME}/bin/mpicc
export MPI_CXX_COMPILER=${MPI_HOME}/bin/mpicxx
export MPI_Fortran_COMPILER=${MPI_HOME}/bin/mpifort
if [ "${MPI}" == "spectrum-mpi" ]; then
    WITH_SPECTRUM=ON
fi

################################################################
# Initialize GPU libraries
################################################################

if [ "${CLUSTER}" == "surface" -o "${CORAL}" -eq 1 -o "${CLUSTER}" == "pascal" ]; then
    HAS_GPU=1
    WITH_CUDA=${WITH_CUDA:-ON}
    WITH_CUDNN=ON
    WITH_CUB=ON
    ELEMENTAL_USE_CUBLAS=OFF
    WITH_ALUMINUM=${WITH_ALUMINUM:-ON}
    ALUMINUM_WITH_NCCL=${ALUMINUM_WITH_NCCL:-ON}
	if [[ ${CORAL} -eq 1 ]]; then
		export NCCL_DIR=/usr/workspace/wsb/brain/nccl2/nccl_2.4.2-1+cuda9.2_ppc64le
		module del cuda
		CUDA_TOOLKIT_MODULE=${CUDA_TOOLKIT_MODULE:-cuda/9.2.148}
	else
		export NCCL_DIR=/usr/workspace/wsb/brain/nccl2/nccl_2.4.2-1+cuda9.2_x86_64
	fi

    # Hack for surface
	case $CLUSTER in
		surface)
		    . /usr/share/[mM]odules/init/bash
		    CUDA_TOOLKIT_MODULE=cudatoolkit/9.2
		    ;;
		pascal)
                    module load opt
		    CUDA_TOOLKIT_MODULE=cudatoolkit/9.2
		    ;;
	esac
fi

if [ "${WITH_CUDA}" == "ON" ]; then
	# Defines CUDA_TOOLKIT_ROOT_DIR
	if [ -z "${CUDA_TOOLKIT_ROOT_DIR}" ]; then
		if [ -n "${CUDA_PATH}" ]; then
			CUDA_TOOLKIT_ROOT_DIR=${CUDA_PATH}
		elif [ -n "${CUDA_HOME}" ]; then
			CUDA_TOOLKIT_ROOT_DIR=${CUDA_HOME}
		elif [ -n "${CUDA_TOOLKIT_MODULE}" -o ${USE_MODULES} -ne 0 ]; then
			CUDA_TOOLKIT_MODULE=${CUDA_TOOLKIT_MODULE:-cuda}
			module load ${CUDA_TOOLKIT_MODULE}
			CUDA_TOOLKIT_ROOT_DIR=${CUDA_HOME:-${CUDA_PATH}}
		fi
	fi
	if [ -n "${CUDA_TOOLKIT_ROOT_DIR}" -a -d "${CUDA_TOOLKIT_ROOT_DIR}" ]; then
		export CUDA_TOOLKIT_ROOT_DIR
	else
		echo "Could not find CUDA"
		exit 1
	fi
	# Defines CUDA_TOOLKIT_VERSION
	#CUDA_TOOLKIT_VERSION=$(ls -l ${CUDA_TOOLKIT_ROOT_DIR} | awk '{print $NF}' | cut -d '-' -f 2)
	CUDA_TOOLKIT_VERSION=$(${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc --version | grep -oE "V[0-9]+\.[0-9]+" | sed 's/V//')

	# CUDNN
	if [ -z "${CUDNN_DIR}" ]; then
		if [ "${CUDA_TOOLKIT_VERSION}" == "9.2" ]; then
			CUDNN_DIR=/usr/workspace/wsb/brain/cudnn/cudnn-7.5.1/cuda-${CUDA_TOOLKIT_VERSION}_${ARCH}
		elif [ "${CUDA_TOOLKIT_VERSION}" == "9.1" ]; then
			CUDNN_DIR=/usr/workspace/wsb/brain/cudnn/cudnn-7.1.3/cuda-${CUDA_TOOLKIT_VERSION}_${ARCH}
		fi
	fi
	if [ ! -d "${CUDNN_DIR}" ]; then
		echo "Could not find cuDNN at $CUDNN_DIR"
		exit 1
	fi
	export CUDNN_DIR
else
    HAS_GPU=0
    WITH_CUDA=${WITH_CUDA:-OFF}
    WITH_CUDNN=OFF
    ELEMENTAL_USE_CUBLAS=OFF
fi

################################################################
# Library options
################################################################
if [ "${CLUSTER}" == "sierra" -o "${CLUSTER}" == "lassen" ]; then
	OPENBLAS_ARCH="TARGET=POWER8"
else
	OPENBLAS_ARCH=
fi

################################################################
# Setup Ninja, if using
################################################################

if [ ${USE_NINJA} -ne 0 ]; then
    if ! which ninja ; then
        if [ "${ARCH}" == "x86_64" ]; then
            export PATH=/usr/workspace/wsb/brain/utils/toss3/ninja/bin:$PATH
        elif [ "${ARCH}" == "ppc64le" ]; then
            export PATH=/usr/workspace/wsb/brain/utils/coral/ninja/bin:$PATH
        fi
    fi
    if ! which ninja ; then
        USE_NINJA=0
    fi
fi
################################################################
# Display parameters
################################################################

# Function to print variable
function print_variable {
    echo "${1}=${!1}"
}

# Print parameters
if [ ${VERBOSE} -ne 0 ]; then
    echo ""
    echo "----------------------"
    echo "System parameters"
    echo "----------------------"
    print_variable CLUSTER
    print_variable TOSS
    print_variable ARCH
    print_variable HAS_GPU
    print_variable USE_MODULES
    echo ""
    echo "----------------------"
    echo "Compiler parameters"
    echo "----------------------"
    print_variable COMPILER
    print_variable COMPILER_VERSION
    print_variable MPI
    print_variable C_COMPILER
    print_variable CXX_COMPILER
    print_variable Fortran_COMPILER
    print_variable MPI_C_COMPILER
    print_variable MPI_CXX_COMPILER
    print_variable MPI_Fortran_COMPILER
    print_variable C_FLAGS
    print_variable CXX_FLAGS
    print_variable Fortran_FLAGS
    echo ""
    echo "----------------------"
    echo "Build parameters"
    echo "----------------------"
    print_variable BUILD_TOOL
    print_variable BUILD_TYPE
    print_variable BUILD_SUFFIX
    print_variable BUILD_DIR
    print_variable INSTALL_DIR
    print_variable Elemental_DIR
    print_variable OpenCV_DIR
    print_variable VTUNE_DIR
    print_variable WITH_CUDA
    print_variable WITH_CUDNN
    print_variable WITH_NVPROF
    print_variable ELEMENTAL_USE_CUBLAS
    print_variable ELEMENTAL_MATH_LIBS
    print_variable PATCH_OPENBLAS
    print_variable DETERMINISTIC
    print_variable CLEAN_BUILD
    print_variable VERBOSE
    print_variable MAKE_NUM_PROCESSES
    print_variable GEN_DOC
    print_variable WITH_TOPO_AWARE
    echo ""
fi

################################################################
# Build LBANN
################################################################

# Work in build directory
pushd ${BUILD_DIR}

# Clean up build directory
if [ ${CLEAN_BUILD} -ne 0 ]; then
    CLEAN_COMMAND="rm -rf ${BUILD_DIR}/*"
    if [ ${VERBOSE} -ne 0 ]; then
        echo "${CLEAN_COMMAND}"
    fi
    eval ${CLEAN_COMMAND}
fi

if [[ ((${BUILD_TOOL} == "make" && -f ${BUILD_DIR}/lbann/build/Makefile) ||
       (${BUILD_TOOL} == "ninja" && -f ${BUILD_DIR}/lbann/build/build.ninja))
      && (${RECONFIGURE} != 1) ]]; then
    echo "Building previously configured LBANN"
    cd ${BUILD_DIR}/lbann/build/
    ${BUILD_TOOL} -j${MAKE_NUM_PROCESSES} all
    ${BUILD_TOOL} install -j${MAKE_NUM_PROCESSES} all
    exit $?
fi

# ATM: goes after Elemental_DIR
#-D OpenCV_DIR=${OpenCV_DIR} \

# Setup the CMake generator
GENERATOR="\"Unix Makefiles\""
if [ ${USE_NINJA} -ne 0 ]; then
    GENERATOR="Ninja"
fi

# Configure build with CMake
CONFIGURE_COMMAND=$(cat << EOF
cmake \
-G ${GENERATOR} \
-D CMAKE_EXPORT_COMPILE_COMMANDS=ON \
-D CMAKE_BUILD_TYPE=${BUILD_TYPE} \
-D CMAKE_INSTALL_MESSAGE=${CMAKE_INSTALL_MESSAGE} \
-D CMAKE_INSTALL_PREFIX=${INSTALL_DIR} \
-D LBANN_SB_BUILD_CEREAL=ON \
-D LBANN_SB_BUILD_CNPY=ON \
-D LBANN_SB_BUILD_HYDROGEN=ON \
-D LBANN_SB_FWD_HYDROGEN_Hydrogen_ENABLE_CUDA=${WITH_CUDA} \
-D LBANN_SB_BUILD_OPENBLAS=ON \
-D LBANN_SB_BUILD_OPENCV=ON \
-D LBANN_SB_BUILD_JPEG_TURBO=ON \
-D LBANN_SB_BUILD_PROTOBUF=ON \
-D LBANN_SB_BUILD_CUB=${WITH_CUB} \
-D LBANN_SB_BUILD_ALUMINUM=${WITH_ALUMINUM} \
-D ALUMINUM_ENABLE_MPI_CUDA=${ALUMINUM_WITH_MPI_CUDA} \
-D ALUMINUM_ENABLE_NCCL=${ALUMINUM_WITH_NCCL} \
-D LBANN_SB_BUILD_CONDUIT=${WITH_CONDUIT} \
-D LBANN_SB_BUILD_HDF5=${WITH_CONDUIT} \
-D LBANN_SB_BUILD_LBANN=ON \
-D CMAKE_CXX_FLAGS="${CXX_FLAGS}" \
-D CMAKE_C_FLAGS="${C_FLAGS}" \
-D CMAKE_C_COMPILER=${C_COMPILER} \
-D CMAKE_CXX_COMPILER=${CXX_COMPILER} \
-D CMAKE_Fortran_COMPILER=${Fortran_COMPILER} \
-D LBANN_WITH_CUDA=${WITH_CUDA} \
-D LBANN_WITH_NVPROF=${WITH_NVPROF} \
-D LBANN_WITH_VTUNE=${WITH_VTUNE} \
-D LBANN_WITH_TBINF=${WITH_TBINF} \
-D LBANN_WITH_TOPO_AWARE=${WITH_TOPO_AWARE} \
-D LBANN_DATATYPE=${DATATYPE} \
-D LBANN_DETERMINISTIC=${DETERMINISTIC} \
-D LBANN_WITH_ALUMINUM=${WITH_ALUMINUM} \
-D LBANN_NO_OMP_FOR_DATA_READERS=${NO_OMP_FOR_DATA_READERS} \
-D LBANN_CONDUIT_DIR=${CONDUIT_DIR} \
-D LBANN_BUILT_WITH_SPECTRUM=${WITH_SPECTRUM} \
-D OPENBLAS_ARCH_COMMAND=${OPENBLAS_ARCH} \
${SUPERBUILD_DIR}
EOF
)


if [ ${VERBOSE} -ne 0 ]; then
    echo "${CONFIGURE_COMMAND}" |& tee cmake_superbuild_invocation.txt
else
    echo "${CONFIGURE_COMMAND}" > cmake_superbuild_invocation.txt
fi
eval ${CONFIGURE_COMMAND}
if [ $? -ne 0 ]; then
    echo "--------------------"
    echo "CONFIGURE FAILED"
    echo "--------------------"
    exit 1
fi

BUILD_OPTIONS="-j${MAKE_NUM_PROCESSES}"
if [ ${VERBOSE} -ne 0 ]; then
  if [ "${BUILD_TOOL}" == "ninja" ]; then
      BUILD_OPTIONS+=" -v"
  else
      BUILD_OPTIONS+=" VERBOSE=${VERBOSE}"
  fi
fi

# Build LBANN with make
# Note: Ensure Elemental to be built before LBANN. Dependency violation appears to occur only when using cuda_add_library.
BUILD_COMMAND="make -j${MAKE_NUM_PROCESSES} VERBOSE=${VERBOSE}"
if [ ${USE_NINJA} -ne 0 ]; then
    if [ ${NINJA_NUM_PROCESSES} -ne 0 ]; then
        BUILD_COMMAND="ninja -j${NINJA_NUM_PROCESSES}"
    else
        # Usually equivalent to -j<num_cpus+2>
        BUILD_COMMAND="ninja"
    fi
fi
if [ ${VERBOSE} -ne 0 ]; then
    echo "${BUILD_COMMAND}"
fi
eval ${BUILD_COMMAND}
if [ $? -ne 0 ]; then
    echo "--------------------"
    echo "BUILD FAILED"
    echo "--------------------"
    exit 1
fi

# Install LBANN with make
if [ ${INSTALL_LBANN} -ne 0 ]; then
    INSTALL_COMMAND="make install -j${MAKE_NUM_PROCESSES} VERBOSE=${VERBOSE}"
    if [ ${USE_NINJA} -ne 0 ]; then
        if [ ${NINJA_NUM_PROCESSES} -ne 0 ]; then
            BUILD_COMMAND="ninja -j${NINJA_NUM_PROCESSES} install"
        else
            # Usually equivalent to -j<num_cpus+2>
            BUILD_COMMAND="ninja install"
        fi
    fi
    if [ ${VERBOSE} -ne 0 ]; then
        echo "${INSTALL_COMMAND}"
    fi
    eval ${INSTALL_COMMAND}
    if [ $? -ne 0 ]; then
        echo "--------------------"
        echo "INSTALL FAILED"
        echo "--------------------"
        exit 1
    fi
fi

# Generate documentation with make
if [ ${GEN_DOC} -ne 0 ]; then
    DOC_COMMAND="make doc"
    if [ ${USE_NINJA} -ne 0 ]; then
        DOC_COMMAND="ninja doc"
    fi
    if [ ${VERBOSE} -ne 0 ]; then
        echo "${DOC_COMMAND}"
    fi
    eval ${DOC_COMMAND}
    if [ $? -ne 0 ]; then
        echo "--------------------"
        echo "BUILDING DOC FAILED"
        echo "--------------------"
        exit 1
    fi
fi

# Return to original directory
dirs -c

#!/bin/bash

# Detect OS version
TOSS=$(uname -r | sed 's/\([0-9][0-9]*\.*\)\-.*/\1/g')
ARCH=$(uname -m)

if [ "${TOSS}" == "3.10.0" ]; then
  if [ "${ARCH}" == "x86_64" ]; then
    module load cmake/3.5.2
  elif [ "${ARCH}" == "ppc64le" ]; then
    module load cmake/3.7.2
  fi
else
  # need to initialize modules on earlier versions of TOSS
  . /usr/share/[mM]odules/init/bash
fi

# Detect the cuda toolkit version loaded or use default
if [ "$CUDA_PATH" == "" ] || [ `basename "$CUDA_PATH"` == "" ] ; then
  # use default
  CUDATOOLKIT_VERSION=8.0
  module load cudatoolkit/$CUDATOOLKIT_VERSION
  if [ -d /opt/cudatoolkit/$CUDATOOLKIT_VERSION ] ; then
      CUDA_TOOLKIT_ROOT_DIR=/opt/cudatoolkit/$CUDATOOLKIT_VERSION
  elif [ -d /opt/cudatoolkit-$CUDATOOLKIT_VERSION ] ; then
      CUDA_TOOLKIT_ROOT_DIR=/opt/cudatoolkit-$CUDATOOLKIT_VERSION
  fi
else
  CUDATOOLKIT_VERSION=`basename "$CUDA_PATH" | sed 's/cudatoolkit-//'`
  CUDA_TOOLKIT_ROOT_DIR=$CUDA_PATH
fi

################################################################
# Default options
################################################################

COMPILER=gnu
if [ "${ARCH}" == "x86_64" ]; then
  MPI=mvapich2
elif [ "${ARCH}" == "ppc64le" ]; then
  MPI=spectrum
fi
COMPILER_CC_NAME_VERSION=
BUILD_TYPE=Release
Elemental_DIR=
if [ "${TOSS}" == "3.10.0" ]; then
  OpenCV_DIR=""
  if [ "${ARCH}" == "x86_64" ]; then
    VTUNE_DIR=/usr/tce/packages/vtune/default
  elif [ "${ARCH}" == "ppc64le" ]; then
    VTUNE_DIR=""
  fi
else
  OpenCV_DIR=/usr/gapps/brain/tools/OpenCV/2.4.13
  VTUNE_DIR=/usr/local/tools/vtune
fi
if [ "${ARCH}" == "x86_64" ]; then
  cuDNN_DIR=/usr/gapps/brain/installs/cudnn/v5
elif [ "${ARCH}" == "ppc64le" ]; then
  cuDNN_DIR=""
fi
ELEMENTAL_MATH_LIBS=
CMAKE_C_FLAGS=
CMAKE_CXX_FLAGS=-DLBANN_SET_EL_RNG
CMAKE_Fortran_FLAGS=
CLEAN_BUILD=0
VERBOSE=0
CMAKE_INSTALL_MESSAGE=LAZY
MAKE_NUM_PROCESSES=$(($(nproc) + 1))
GEN_DOC=0
INSTALL_LBANN=0
ALTERNATE_BUILD_DIR=none
BUILD_SUFFIX=
SEQ_INIT=OFF

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
  ${C}--verbose${N}               Verbose output.
  ${C}--debug${N}                 Build with debug flag.
  ${C}--tbinf${N}                 Build with Tensorboard interface.
  ${C}--vtune${N}                 Build with VTune profiling libraries.
  ${C}--clean-build${N}           Clean build directory before building.
  ${C}--make-processes${N} <val>  Number of parallel processes for make.
  ${C}--doc${N}                   Generate documentation.
  ${C}--install-lbann${N}         Install LBANN headers and dynamic library into the build directory.
  ${C}--build${N}                 Specify alternative build directory; default is <lbann_home>/build.
  ${C}--suffix${N}                Specify suffix for build directory. If you are, e.g, building on surface, your build will be <someplace>/surface.llnl.gov, regardless of your choice of compiler or other flags. This option enables you to specify, e.g: --suffix gnu_debug, in which case your build will be in the directory <someplace>/surface.llnl.gov.gnu_debug
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
        ALTERNATE_BUILD_DIR=${2}
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
    -v|--verbose)
      # Verbose output
      VERBOSE=1
      CMAKE_INSTALL_MESSAGE=ALWAYS
      ;;
    -d|--debug)
      # Debug mode
      BUILD_TYPE=Debug
      SEQ_INIT=ON
      ;;
    --tbinf)
      # Tensorboard interface
      WITH_TBINF=ON
      ;;
    --vtune)
      # VTune libraries
      WITH_VTUNE=ON
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
# Load modules
################################################################

if [ "${TOSS}" != "3.10.0" ]; then
  module load git
fi

################################################################
# Initialize variables
################################################################

# Detect computing system
CLUSTER=$(hostname | sed 's/\([a-zA-Z][a-zA-Z]*\)[0-9]*/\1/g')

# Build and install directories
ROOT_DIR=$(git rev-parse --show-toplevel)
if [ "${ALTERNATE_BUILD_DIR}" !=  "none" ]; then
  BUILD_DIR=${ALTERNATE_BUILD_DIR}/${CLUSTER}.llnl.gov
else
  BUILD_DIR=${ROOT_DIR}/build/${CLUSTER}.llnl.gov
fi
INSTALL_DIR=${BUILD_DIR}

if [ -n "${BUILD_SUFFIX}" ]; then
  BUILD_DIR=${BUILD_DIR}.${BUILD_SUFFIX} 
  INSTALL_DIR=${BUILD_DIR}
fi

mkdir -p ${BUILD_DIR}
mkdir -p ${INSTALL_DIR}

# Get C/C++/Fortran compilers and corresponding top-level MPI directory
if [ "${COMPILER}" == "gnu" ]; then
  # GNU compilers
  if [ "${TOSS}" == "3.10.0" ]; then
    if [ "${ARCH}" == "x86_64" ]; then
      GNU_DIR=/usr/tce/packages/gcc/gcc-4.9.3/bin
      GFORTRAN_LIB=/usr/tce/packages/gcc/gcc-4.9.3/lib64/libgfortran.so
      if [ "${MPI}" == "mvapich2" ]; then
        MPI_DIR=/usr/tce/packages/mvapich2/mvapich2-2.2-gcc-4.9.3
      elif [ "${MPI}" == "openmpi" ]; then
        MPI_DIR=/usr/tce/packages/openmpi/openmpi-2.0.0-gcc-4.9.3
      else
        echo "Invalid MPI library selected ${MPI}"
        exit 1
      fi
    elif [ "${ARCH}" == "ppc64le" ]; then
      GNU_DIR=/usr/tcetmp/packages/gcc/gcc-4.9.3/bin
      GFORTRAN_LIB=/usr/tcetmp/packages/gcc/gcc-4.9.3/lib64/libgfortran.so
      if [ "${MPI}" == "spectrum" ]; then
        MPI_DIR=/opt/ibm/spectrum_mpi
      else
        echo "Invalid MPI library selected ${MPI}"
        exit 1
      fi
    fi
  else
    GNU_DIR=/opt/rh/devtoolset-3/root/usr/bin
    #GFORTRAN_LIB=/opt/rh/devtoolset-2/root/usr/lib/gcc/x86_64-redhat-linux/4.8.2/libgfortran.so
    GFORTRAN_LIB=/opt/rh/devtoolset-3/root/usr/lib/gcc/x86_64-redhat-linux/4.9.2
    MPI_DIR=/usr/local/tools/mvapich2-gnu-2.1
  fi
  CMAKE_C_COMPILER=${GNU_DIR}/gcc
  CMAKE_CXX_COMPILER=${GNU_DIR}/g++
  CMAKE_Fortran_COMPILER=${GNU_DIR}/gfortran

#  COMPILER_VERSION=
#  COMPILER_CC_NAME_VERSION=gcc

elif [ "${COMPILER}" == "intel" ]; then
  # Intel compilers
  if [ "${TOSS}" == "3.10.0" ]; then
    INTEL_DIR=/usr/tce/packages/intel/intel-17.0.2/bin
    MPI_DIR=/usr/tce/packages/mvapich2/mvapich2-2.2-intel-17.0.2
  else
    INTEL_DIR=/opt/intel-16.0/linux/bin/intel64
    MPI_DIR=/usr/local/tools/mvapich2-intel-2.1
  fi
  CMAKE_C_COMPILER=${INTEL_DIR}/icc
  CMAKE_CXX_COMPILER=${INTEL_DIR}/icpc
  CMAKE_Fortran_COMPILER=${INTEL_DIR}/ifort

elif [ "${COMPILER}" == "clang" ]; then
  if [ "${TOSS}" == "3.10.0" ]; then
    # (dah) clang is not currently installed on catalyst, so this will fail ...
    GNU_DIR=/usr/global/tools/clang/chaos_5_x86_64_ib/clang-3.8.0/bin
    GFORTRAN_LIB=/usr/tce/packages/gcc/gcc-4.9.3/lib64/libgfortran.so
    MPI_DIR=/usr/tce/packages/mvapich2/mvapich2-2.2-gcc-4.9.3
  else
    GNU_DIR=/usr/global/tools/clang/chaos_5_x86_64_ib/clang-3.8.0/bin
    GFORTRAN_LIB=/opt/rh/devtoolset-2/root/usr/lib/gcc/x86_64-redhat-linux/4.8.2/libgfortran.so
    MPI_DIR=/usr/local/tools/mvapich2-gnu-2.1
  fi
  CMAKE_CXX_FLAGS="-DLBANN_SET_EL_RNG"
  CMAKE_C_COMPILER=/usr/global/tools/clang/chaos_5_x86_64_ib/clang-3.8.0/bin/clang
  CMAKE_CXX_COMPILER=/usr/global/tools/clang/chaos_5_x86_64_ib/clang-3.8.0/bin/clang++
  CMAKE_Fortran_COMPILER=/opt/rh/devtoolset-3/root/usr/bin/gfortran

else
  # Unrecognized compiler
  echo "Unrecognized compiler (${COMPILER})"
  exit 1
fi

# Get MPI compilers
MPI_C_COMPILER=${MPI_DIR}/bin/mpicc
MPI_CXX_COMPILER=${MPI_DIR}/bin/mpicxx
MPI_Fortran_COMPILER=${MPI_DIR}/bin/mpifort

# Get CUDA and cuDNN
if [ "${CLUSTER}" == "surface" ]; then
  WITH_CUDA=ON
  WITH_CUDNN=ON
  ELEMENTAL_USE_CUBLAS=1
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
    ${CLEAN_COMMAND}
  fi

# ATM: goes after Elemental_DIR
#-D OpenCV_DIR=${OpenCV_DIR} \

  # Configure build with CMake
  CONFIGURE_COMMAND=$(cat << EOF
cmake \
-D CMAKE_BUILD_TYPE=${BUILD_TYPE} \
-D CMAKE_INSTALL_MESSAGE=${CMAKE_INSTALL_MESSAGE} \
-D CMAKE_INSTALL_PREFIX=${INSTALL_DIR} \
-D CMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} \
-D CMAKE_C_COMPILER=${CMAKE_C_COMPILER} \
-D CMAKE_Fortran_COMPILER=${CMAKE_Fortran_COMPILER} \
-D GFORTRAN_LIB=${GFORTRAN_LIB} \
-D MPI_CXX_COMPILER=${MPI_CXX_COMPILER} \
-D MPI_C_COMPILER=${MPI_C_COMPILER} \
-D MPI_Fortran_COMPILER=${MPI_Fortran_COMPILER} \
-D CMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS} \
-D CMAKE_C_FLAGS=${CMAKE_C_FLAGS} \
-D CMAKE_Fortran_FLAGS=${CMAKE_Fortran_FLAGS} \
-D WITH_CUDA=${WITH_CUDA} \
-D WITH_CUDNN=${WITH_CUDNN} \
-D WITH_TBINF=${WITH_TBINF} \
-D WITH_VTUNE=${WITH_VTUNE} \
-D Elemental_DIR=${Elemental_DIR} \
-D CUDA_TOOLKIT_ROOT_DIR=${CUDA_TOOLKIT_ROOT_DIR} \
-D cuDNN_DIR=${cuDNN_DIR} \
-D VTUNE_DIR=${VTUNE_DIR} \
-D ELEMENTAL_MATH_LIBS=${ELEMENTAL_MATH_LIBS} \
-D VERBOSE=${VERBOSE} \
-D MAKE_NUM_PROCESSES=${MAKE_NUM_PROCESSES} \
-D LBANN_HOME=${ROOT_DIR} \
-D SEQ_INIT=${SEQ_INIT} \
-D ELEMENTAL_USE_CUBLAS=${ELEMENTAL_USE_CUBLAS}
${ROOT_DIR}
EOF
)

  if [ ${VERBOSE} -ne 0 ]; then
    echo "${CONFIGURE_COMMAND}"
  fi
  ${CONFIGURE_COMMAND}
  if [ $? -ne 0 ] ; then
    echo "--------------------"
    echo "CONFIGURE FAILED"
    echo "--------------------"
    exit 1
  fi

  # Build LBANN with make
  BUILD_COMMAND="make -j${MAKE_NUM_PROCESSES} VERBOSE=${VERBOSE}"
  if [ ${VERBOSE} -ne 0 ]; then
    echo "${BUILD_COMMAND}"
  fi
  ${BUILD_COMMAND}
  if [ $? -ne 0 ] ; then
    echo "--------------------"
    echo "MAKE FAILED"
    echo "--------------------"
    exit 1
  fi

  # Install LBANN with make
  if [ ${INSTALL_LBANN} -ne 0 ]; then
      INSTALL_COMMAND="make install -j${MAKE_NUM_PROCESSES} VERBOSE=${VERBOSE}"
      if [ ${VERBOSE} -ne 0 ]; then
          echo "${INSTALL_COMMAND}"
      fi
      ${INSTALL_COMMAND}
      if [ $? -ne 0 ] ; then
          echo "--------------------"
          echo "MAKE INSTALL FAILED"
          echo "--------------------"
          exit 1
      fi
  fi

  # Generate documentation with make
  if [ ${GEN_DOC} -ne 0 ]; then
    DOC_COMMAND="make doc"
    if [ ${VERBOSE} -ne 0 ]; then
      echo "${DOC_COMMAND}"
    fi
    ${DOC_COMMAND}
    if [ $? -ne 0 ] ; then
      echo "--------------------"
      echo "MAKE DOC FAILED"
      echo "--------------------"
      exit 1
    fi
  fi
  
popd

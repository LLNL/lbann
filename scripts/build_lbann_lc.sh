#!/bin/bash

################################################################
# Default options
################################################################

COMPILER=gnu
BUILD_TYPE=Release
Elemental_DIR=
OpenCV_DIR=/usr/gapps/brain/tools/OpenCV/2.4.13
CUDA_TOOLKIT_ROOT_DIR=/opt/cudatoolkit-7.5
cuDNN_DIR=/usr/gapps/brain/installs/cudnn/v5
ELEMENTAL_MATH_LIBS=
CMAKE_C_FLAGS=
CMAKE_CXX_FLAGS=
CMAKE_Fortran_FLAGS=
CLEAN_BUILD=0
VERBOSE=0
CMAKE_INSTALL_MESSAGE=LAZY
MAKE_NUM_PROCESSES=$(($(nproc) + 1))

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
  ${C}--compiler${N} <val>        Specify compiler ('gnu' or 'intel').
  ${C}--verbose${N}               Verbose output.
  ${C}--debug${N}                 Build with debug flag.
  ${C}--tbinf${N}                 Build with Tensorboard interface.
  ${C}--clean-build${N}           Clean build directory before building.
  ${C}--make-processes${N} <val>  Number of parallel processes for make.
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
    --compiler)
      # Choose compiler
      if [ -n "${2}" ]; then
        COMPILER=${2}
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
      ;;
    --tbinf)
      # Tensorboard interface
      WITH_TBINF=ON
      ;;
    --clean-build|--build-clean)
      # Clean build directory
      CLEAN_BUILD=1
      ;;
    -j|--make-processes)
      if [ -n "${2}" ]; then
        MAKE_NUM_PROCESSES=${2}
      else
        echo "\"${1}\" option requires a non-empty option argument" >&2
        exit 1
      fi
      ;;
    -?*)
      # Unknown option
      echo "Ignored unknown option (${1})" >&2
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

module load git
module load cudatoolkit/7.5

################################################################
# Initialize variables
################################################################

# Detect computing system
CLUSTER=$(hostname | sed 's/\([a-zA-Z][a-zA-Z]*\)[0-9]*/\1/g')

# Build and install directories
ROOT_DIR=$(git rev-parse --show-toplevel)
BUILD_DIR=${ROOT_DIR}/build/${CLUSTER}.llnl.gov
INSTALL_DIR=${BUILD_DIR}
mkdir -p ${BUILD_DIR}
mkdir -p ${INSTALL_DIR}

# Get C/C++/Fortran compilers
if [ "${COMPILER}" == "gnu" ]; then
  # GNU compilers
  GNU_DIR=/opt/rh/devtoolset-3/root/usr/bin
  CMAKE_C_COMPILER=${GNU_DIR}/gcc
  CMAKE_CXX_COMPILER=${GNU_DIR}/g++
  CMAKE_Fortran_COMPILER=${GNU_DIR}/gfortran
  GFORTRAN_LIB=/opt/rh/devtoolset-2/root/usr/lib/gcc/x86_64-redhat-linux/4.8.2/libgfortran.so
elif [ "${COMPILER}" == "intel" ]; then
  # Intel compilers
  INTEL_DIR=/opt/intel-16.0/linux/bin/intel64
  CMAKE_C_COMPILER=${INTEL_DIR}/icc
  CMAKE_CXX_COMPILER=${INTEL_DIR}/icpc
  CMAKE_Fortran_COMPILER=${INTEL_DIR}/ifort
else
  # Unrecognized compiler
  echo "Unrecognized compiler (${COMPILER})"
  exit 1
fi

# Get MPI compilers
MPI_DIR=/usr/local/tools/mvapich2-${COMPILER}-2.1
MPI_C_COMPILER=${MPI_DIR}/bin/mpicc
MPI_CXX_COMPILER=${MPI_DIR}/bin/mpicxx
MPI_Fortran_COMPILER=${MPI_DIR}/bin/mpifort

# Get CUDA and cuDNN
if [ "${CLUSTER}" == "surface" ]; then
  WITH_CUDA=ON
  WITH_CUDNN=ON
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
-D Elemental_DIR=${Elemental_DIR} \
-D OpenCV_DIR=${OpenCV_DIR} \
-D CUDA_TOOLKIT_ROOT_DIR=${CUDA_TOOLKIT_ROOT_DIR} \
-D cuDNN_DIR=${cuDNN_DIR} \
-D ELEMENTAL_MATH_LIBS=${ELEMENTAL_MATH_LIBS} \
-D VERBOSE=${VERBOSE} \
-D MAKE_NUM_PROCESSES=${MAKE_NUM_PROCESSES} \
${ROOT_DIR}
EOF
)
  if [ ${VERBOSE} -ne 0 ]; then
    echo "${CONFIGURE_COMMAND}"
  fi
  ${CONFIGURE_COMMAND}
  rc=$?; if [[ $rc != 0 ]]; then exit $rc; fi

  # Build LBANN with make
  BUILD_COMMAND="make -j${MAKE_NUM_PROCESSES} VERBOSE=${VERBOSE}"
  if [ ${VERBOSE} -ne 0 ]; then
    echo "${BUILD_COMMAND}"
  fi
  ${BUILD_COMMAND}
  rc=$?; if [[ $rc != 0 ]]; then exit $rc; fi

  # Build LBANN with make
  INSTALL_COMMAND="make install -j${MAKE_NUM_PROCESSES} VERBOSE=${VERBOSE}"
  if [ ${VERBOSE} -ne 0 ]; then
    echo "${INSTALL_COMMAND}"
  fi
  ${INSTALL_COMMAND}
  rc=$?; if [[ $rc != 0 ]]; then exit $rc; fi
  
popd

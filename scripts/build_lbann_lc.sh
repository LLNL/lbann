#!/bin/bash

# Default options
COMPILER=gnu

# Help message
function help_message {
  local n=`tput sgr0`    # Normal text
  local c=`tput setf 4`  # Colored text
cat << EOF
Build LBANN on an LLNL LC system.
Usage: ${SCRIPT} [options]
Options:
  ${C}--help${N}                  Display this help message and exit.
  ${C}--compiler${N} <val>        Specify compiler ('gnu' or 'intel').
  ${C}--verbose${N}               Verbose output.
  ${C}--clean-build${N}           Clean build directory before building.
  ${C}--make-processes${N} <val>  Number of parallel processes for make.
EOF
}

# Parse command-line arguments
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
      ;;
    --build-clean)
      # Clean build directory
      BUILD_CLEAN=1
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

# Detect computing system
HOSTNAME=$(hostname)
if [[ "${HOSTNAME}" == *"surface"* ]]; then
  CLUSTER="surface"
elif [[ "${HOSTNAME}" == *"catalyst"* ]]; then
  CLUSTER="catalyst"
elif [[ "${HOSTNAME}" == *"flash"* ]]; then
  CLUSTER="flash"
elif [[ "${HOSTNAME}" == *"vandamme"* ]]; then
  CLUSTER="vandamme"
else
  echo "Unrecognized host name (${HOSTNAME})"
  exit 1
fi

# Build and install directories
BUILD_DIR=$(git rev-parse --show-toplevel)/build/${CLUSTER}.llnl.gov
INSTALL_DIR=${BUILD_DIR}

# Get C/C++/Fortran compilers
if [ "${COMPILER}" == "gnu" ]; then
  # GNU compilers
  GNU_DIR=/opt/rh/devtoolset-2/root/usr/bin
  CMAKE_C_COMPILER=${GNU_DIR}/gcc
  CMAKE_CXX_COMPILER=${GNU_DIR}/g++
  CMAKE_Fortran_COMPILER=${GNU_DIR}/gfortran
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
MPI_DIR=/usr/local/tools/mvapich2-${COMPILER}-2.1/bin
MPI_C_COMPILER=${MPI_DIR}/mpicc
MPI_CXX_COMPILER=${MPI_DIR}/mpicxx
MPI_Fortran_COMPILER=${MPI_DIR}/mpifort

# Get CUDA and cuDNN
if [ "${CLUSTER}" == "surface" ]; then
  CUDA_TOOLKIT_ROOT_DIR="/opt/cudatoolkit-7.5"
  CMAKE_CUDNN_DIR="/usr/gapps/brain/installs/cudnn/v5"
fi

# Number of parallel processes for make
if [ -z "${MAKE_NUM_PROCESSES}" ]; then
  MAKE_NUM_PROCESSES=$(($(nproc) + 1))
fi

# Work in build directory
pushd ${BUILD_DIR}

  # Clean up build directory
  if [ -n "${BUILD_CLEAN}" ]; then
    rm -rf *
  fi

  # Initialize build with CMake
  cmake  \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_INSTALL_PREFIX=${INSTALL_DIR} \
    -D CMAKE_C_COMPILER=${CMAKE_C_COMPILER} \
    -D CMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} \
    -D CMAKE_Fortran_COMPILER=${CMAKE_Fortran_COMPILER} \
    -D MPI_C_COMPILER=${MPI_C_COMPILER} \
    -D MPI_CXX_COMPILER=${MPI_CXX_COMPILER} \
    -D MPI_Fortran_COMPILER=${MPI_Fortran_COMPILER} \
    -D MATH_LIBS="-L/usr/gapps/brain/installs/BLAS/${CLUSTER}/lib -lopenblas" \
    -D OpenCV_DIR="/usr/gapps/brain/tools/OpenCV/2.4.13/" \
    -D CUDA_TOOLKIT_ROOT_DIR=${CUDA_TOOLKIT_ROOT_DIR} \
    -D CMAKE_CUDNN_DIR=${CMAKE_CUDNN_DIR} \
    -D MAKE_NUM_PROCESSES=${MAKE_NUM_PROCESSES} \
    ../..

  # Compile LBANN
  make -j${MAKE_NUM_PROCESSES} VERBOSE=${VERBOSE}

popd

#!/bin/bash

################################################################
# Simple script to build LBANN in OS X.
# Can be called anywhere in the LBANN project tree.
################################################################

brew_check_install()
{
    PKG=$(brew list | grep $1)
    if [ -z "$PKG" ]; then
        brew install $1
    fi
}

# Install dependencies with Homebrew
# Note: Requires sudo access. Homebrew can be downloaded from
# http://brew.sh
# and installed / updated as shown below
#   brew tap homebrew/science
#   brew tap homebrew/boneyard
#   brew update
# Check for LBANN dependencies
brew_check_install git
brew_check_install cmake
brew_check_install clang-omp  # Require OpenMP support in clang
brew_check_install gcc49      # gfortran-4.9 is compatible with clang
brew_check_install open-mpi
brew_check_install opencv
brew_check_install doxygen
brew_check_install graphviz   # Doxygen dependency
brew_check_install metis      # Elemental dependency
brew_check_install scalapack  # Elemental dependency

# Parameters
CMAKE_C_COMPILER=/usr/local/bin/clang-omp
CMAKE_CXX_COMPILER=/usr/local/bin/clang-omp++
CMAKE_Fortran_COMPILER=/usr/local/bin/gfortran-4.9
MPI_C_COMPILER=/usr/local/bin/mpicc
MPI_CXX_COMPILER=/usr/local/bin/mpicxx
MPI_Fortran_COMPILER=/usr/local/bin/mpifort
Elemental_DIR=
OpenCV_DIR=/usr/local/share/OpenCV
CUDA_TOOLKIT_ROOT_DIR=
cuDNN_DIR=
VERBOSE=1
MAKE_NUM_PROCESSES=$(($(sysctl -n hw.ncpu) + 1))

# Build and install directories
ROOT_DIR=$(git rev-parse --show-toplevel)
BUILD_DIR=${ROOT_DIR}/build/$(hostname)
INSTALL_DIR=${BUILD_DIR}
mkdir -p ${BUILD_DIR}
mkdir -p ${INSTALL_DIR}

# Work in build directory
pushd ${BUILD_DIR}

  # Clear build directory
  rm -rf *

  # Configure build with CMake
  cmake \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_INSTALL_MESSAGE=LAZY \
    -D CMAKE_INSTALL_PREFIX=${INSTALL_DIR} \
    -D CMAKE_C_COMPILER=${CMAKE_C_COMPILER} \
    -D CMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} \
    -D CMAKE_Fortran_COMPILER=${CMAKE_Fortran_COMPILER} \
    -D MPI_C_COMPILER=${MPI_C_COMPILER} \
    -D MPI_CXX_COMPILER=${MPI_CXX_COMPILER} \
    -D MPI_Fortran_COMPILER=${MPI_Fortran_COMPILER} \
    -D Elemental_DIR=${Elemental_DIR} \
    -D OpenCV_DIR=${OpenCV_DIR} \
    -D CUDA_TOOLKIT_ROOT_DIR=${CUDA_TOOLKIT_ROOT_DIR} \
    -D cuDNN_DIR=${cuDNN_DIR} \
    -D WITH_TBINF=OFF \
    -D VERBOSE=${VERBOSE} \
    -D MAKE_NUM_PROCESSES=${MAKE_NUM_PROCESSES} \
    ${ROOT_DIR}

  # Build LBANN with make
  make -j${MAKE_NUM_PROCESSES} VERBOSE=${VERBOSE}

  # Install LBANN with make
  make install -j${MAKE_NUM_PROCESSES} VERBOSE=${VERBOSE}
  
popd

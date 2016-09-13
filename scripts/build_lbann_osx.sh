#!/bin/bash

################################################################
# Simple script to build LBANN in OS X.
# Can be called anywhere in the LBANN project tree.
# TODO: fix linking errors with OpenCV
################################################################

# Install dependencies with Homebrew
# Note: Requires sudo access. Homebrew can be downloaded from
# http://brew.sh
brew tap homebrew/science
brew update
brew install git
brew install cmake
brew install gcc49
brew install mpich
brew install opencv
brew install doxygen
brew install metis

# Parameters
CMAKE_C_COMPILER=/usr/bin/clang
CMAKE_CXX_COMPILER=/usr/bin/clang++
CMAKE_Fortran_COMPILER=/usr/local/bin/gfortran-4.9
MPI_C_COMPILER=/usr/local/bin/mpicc
MPI_CXX_COMPILER=/usr/local/bin/mpicxx
MPI_Fortran_COMPILER=/usr/local/bin/mpifort
Elemental_DIR=
OpenCV_DIR=
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

popd

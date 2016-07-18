#!/bin/bash

COMPILER=gnu

#INSTALL_PREFIX=/usr/gapps/brain/tools/Elemental/El_0.85/intel_cc
INSTALL_PREFIX=${PWD}

EL_VER=El_0.86/v86-4748937
ELEMENTAL_DIR=/usr/gapps/brain/tools/Elemental/${EL_VER}/${COMPILER}_cc

#include /usr/gapps/brain/tools/Elemental/${EL_VER}/${COMPILER}/conf/ElVars

BLAS=/usr/gapps/brain/installs/BLAS/catalyst
CC_PATH=/opt/rh/devtoolset-2/root/usr/bin
CC_LIBRARY_PATH=${BLAS}/lib
#MATHLIBS="-L${BLAS}/lib -lopenblas"
MPI_PATH=/usr/local/tools/mvapich2-${COMPILER}-2.2/bin

rm CMakeCache.txt

cmake ../../ \
  -DCMAKE_CXX_COMPILER=${CC_PATH}/g++ \
  -DCMAKE_C_COMPILER=${CC_PATH}/gcc \
  -DCMAKE_LIBRARY_PATH=${CC_LIBRARY_PATH} \
  -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
  -DCMAKE_BUILD_TYPE=Release \
  -DMPI_C_COMPILER=${MPI_PATH}/mpicc \
  -DMPI_CXX_COMPILER=${MPI_PATH}/mpicxx \
  -DOpenCV_DIR="/usr/gapps/brain/tools/OpenCV/2.4.13/" \
  -DCMAKE_ELEMENTAL_DIR=${ELEMENTAL_DIR} \
  -DElemental_ROOT_DIR=${ELEMENTAL_DIR} \
  -DOpenBLAS_DIR=${BLAS} \
  -DCMAKE_CXX_FLAGS_RELEASE="-std=c++11 -O3" \
  -DCUDA_TOOLKIT_ROOT_DIR="/opt/cudatoolkit-7.5" \
  -DCMAKE_CUDNN_DIR="/usr/gapps/brain/installs/cudnn/v5" \


#  -DCMAKE_BLAS_ROOT_DIR=${BLAS} \
#  -DCMAKE_CXX_FLAGS_RELEASE="-std=c++11 -O3 -L${BLAS}/lib -lopenblas" \
#   -DCMAKE_OPENCV_DIR="/usr/gapps/brain/installs/generic/share/OpenCV" \
#  -DCMAKE_ELEMENTAL_DIR=${ELEMENTAL_DIR} \
#  -DMATH_LIBS="-mkl" \
#  -DINSTALL_PYTHON_PACKAGE=OFF
#  -DEL_DISABLE_METIS=TRUE \
#  -DPYTHON_SITE_PACKAGES=/usr/gapps/brain/installs/generic/lib/python2.7/site-packages \


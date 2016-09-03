#!/bin/bash

HOMEBREW_PATH=/usr/local

INSTALL_PREFIX=${PWD}

#include /usr/gapps/brain/tools/Elemental/${EL_VER}/${COMPILER}/conf/ElVars
#CV_INCPATH = -I/usr/gapps/brain/installs/generic/include
#CV_LIBPATH = -L/usr/gapps/brain/installs/generic/lib
#CV_LIBS = -lopencv_core -lopencv_highgui
#CV_PREPROC = -D__LIB_OPENCV

rm CMakeCache.txt
cmake ../../ \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DCMAKE_C_COMPILER=/usr/bin/gcc \
  -DCMAKE_Fortran_COMPILER=/usr/bin/gfortran \
  -DMPI_C_COMPILER=${HOMEBREW_PATH}/bin/mpicc \
  -DMPI_CXX_COMPILER=${HOMEBREW_PATH}/bin/mpic++ \
  -DMPI_CXX_COMPILER=${HOMEBREW_PATH}/bin/mpifortran \
  -DCMAKE_OPENCV_DIR="/usr/local/Cellar/opencv/2.4.12_2" \
  -DCMAKE_CXX_FLAGS_RELEASE="-std=c++11 -O3 -DEL_NEW_MPI_REQUEST" \
  -DMPIEXEC_PREFLAGS="-hosts;localhost" \
  -DElemental_ROOT_DIR=/Users/vanessen1/Research/DeepLearning/tools

#   -DBOOST_ROOT=${HOMEBREW_PATH}/include \

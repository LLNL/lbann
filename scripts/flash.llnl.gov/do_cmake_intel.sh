#!/bin/bash

#INSTALL_PREFIX=/usr/gapps/brain/tools/Elemental/El_0.85/intel_cc
INSTALL_PREFIX=${PWD}

#module load intel/14.0
#module load mkl/14.0
#INCPATH="-I /usr/gapps/brain/tools/Elemental/El_0.86/v86-6ec56a/intel_cc/include/"
#GVER=GIT_VERSION
#PREPROC="-D__LIB_ELEMENTAL"

rm CMakeCache.txt

cmake ../../ \
  -DCMAKE_CXX_COMPILER=/opt/intel/16.0/bin/intel64/icpc \
  -DCMAKE_C_COMPILER=/opt/intel/16.0/bin/intel64/icc \
  -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
  -DCMAKE_BUILD_TYPE=Release \
  -DMPI_C_COMPILER=/opt/openmpi/1.10/intel/bin/mpicc \
  -DMPI_CXX_COMPILER=/opt/openmpi/1.10/intel/bin/mpicxx \
  -DCMAKE_OPENCV_DIR="/usr/gapps/brain/install/generic" \
  -DCMAKE_CXX_FLAGS_RELEASE="-std=c++11 -O3" \
  -DMATH_LIBS="-mkl" \


#  -DINSTALL_PYTHON_PACKAGE=OFF
#  -DEL_DISABLE_METIS=TRUE \
#  -DPYTHON_SITE_PACKAGES=/usr/gapps/brain/installs/generic/lib/python2.7/site-packages \


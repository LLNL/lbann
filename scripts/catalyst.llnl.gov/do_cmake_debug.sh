#!/bin/bash

#INSTALL_PREFIX=/usr/gapps/brain/tools/Elemental/El_0.85/intel_cc
INSTALL_PREFIX=${PWD}

EL_VER=El_0.86/v86-6ec56a
COMPILER=intel_cc-debug
ELEMENTAL_DIR=/usr/gapps/brain/tools/Elemental/${EL_VER}/${COMPILER}

#include /usr/gapps/brain/tools/Elemental/${EL_VER}/${COMPILER}/conf/ElVars

#module load intel/14.0
#module load mkl/14.0
#INCPATH="-I /usr/gapps/brain/tools/Elemental/El_0.86/v86-6ec56a/intel_cc/include/"
#GVER=GIT_VERSION
#PREPROC="-D__LIB_ELEMENTAL"

export INTEL_LICENSE_FILE=/opt/modules/etc/license.client.intel
ICC_VER=14.0
INTEL_LIBRARY_PATH=/opt/intel-${ICC_VER}/compiler/lib/intel64:/opt/intel-${ICC_VER}/mkl/lib/intel64
#export MANPATH=/opt/intel-${ICC_VER}/man:${MANPATH}

rm CMakeCache.txt

cmake ../../ \
  -DCMAKE_CXX_COMPILER=/opt/intel-${ICC_VER}/bin/intel64/icpc \
  -DCMAKE_C_COMPILER=/opt/intel-${ICC_VER}/bin/intel64/icc \
  -DCMAKE_LIBRARY_PATH=${INTEL_LIBRARY_PATH} \
  -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
  -DCMAKE_BUILD_TYPE=Debug \
  -DMPI_C_COMPILER=/usr/local/tools/mvapich2-intel-2.1/bin/mpicc \
  -DMPI_CXX_COMPILER=/usr/local/tools/mvapich2-intel-2.1/bin/mpicxx \
  -DCMAKE_OPENCV_DIR="/usr/gapps/brain/install/generic" \
  -DCMAKE_ELEMENTAL_DIR=${ELEMENTAL_DIR} \
  -DCMAKE_CXX_FLAGS_DEBUG="-std=c++11 -g -O0 -mkl" \


#  -DMATH_LIBS="-mkl" \
#  -DINSTALL_PYTHON_PACKAGE=OFF
#  -DEL_DISABLE_METIS=TRUE \
#  -DPYTHON_SITE_PACKAGES=/usr/gapps/brain/installs/generic/lib/python2.7/site-packages \


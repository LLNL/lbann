#!/bin/sh

thisCommand=`basename $0`
PATCH_DIR=`dirname $0`
OpenBLAS_PATCH_OLD=${PATCH_DIR}/old/OpenBLAS.patch
OpenBLAS_PATCH=${PATCH_DIR}/OpenBLAS.patch

WRONG_OPT=`grep malign-power Makefile.power`
EXTRA_ARCH=`grep I6500 Makefile.system`

if [ "${WRONG_OPT}" != "" ] ; then
  if [ "${EXTRA_ARCH}" != "" ] ; then
    patch -N -s -p0 < ${OpenBLAS_PATCH}
    echo "Applying patch to OpenBLAS branch by Tim"
  else
    patch -N -s -p0 < ${OpenBLAS_PATCH_OLD}
    echo "Applying patch to OpenBLAS source"
  fi
fi

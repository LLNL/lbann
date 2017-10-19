#!/bin/sh

thisCommand=`basename $0`
PATCH_DIR=`dirname $0`
OpenBLAS_PATCH_OLD=${PATCH_DIR}/old/EL_OpenBLAS.patch
OpenBLAS_PATCH=${PATCH_DIR}/EL_OpenBLAS.patch

fOrg=`grep OpenBLAS.cmake ${PATCH_DIR}/ChangedFiles.EL_OpenBLAS.txt`

OpenBLAS_TAG=`grep GIT_TAG ${fOrg} | grep -v OPENBLAS_TAG | awk '{if (NF==2) print $2; else print "0";}' | sed -e 's/\"//g'`

if [ "${OpenBLAS_TAG}" = "" ] ; then
  OpenBLAS_TAG=`grep 'OPENBLAS_TAG' ${PATCH_DIR}/../../cmake/Elemental.cmake | grep set | awk '{if (NF==2) print $2; else print "0";}' | sed -e 's/\"//g' | sed -e 's/)//g'`
fi
echo ${OpenBLAS_TAG}

if [ "${OpenBLAS_TAG}" = "v0.2.15" ] ; then
  patch -N -s -p0 < ${OpenBLAS_PATCH_OLD}
  echo "Set to upgrade OpenBLAS"
elif [ "${OpenBLAS_TAG}" = "db72ad8f6a430393d5137527e296e17b1b1fe5bf" ] ; then
  patch -N -s -p0 < ${OpenBLAS_PATCH}
  echo "Set to patch OpenBLAS"
fi

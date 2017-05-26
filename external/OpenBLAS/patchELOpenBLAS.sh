#!/bin/sh

thisCommand=`basename $0`
PATCH_DIR=`dirname $0`
OpenBLAS_PATCH=${PATCH_DIR}/EL_OpenBLAS.patch

fOrg=`grep OpenBLAS.cmake ${PATCH_DIR}/ChangedFiles.EL_OpenBLAS.txt`

OpenBLAS_TAG=`grep GIT_TAG ${fOrg} | awk '{if (NF==2) print $2; else print "0";}' | sed -e 's/\"//g'`
#echo ${OpenBLAS_TAG}

if [ "${OpenBLAS_TAG}" = "v0.2.15" ] ; then
  patch -p0 < ${OpenBLAS_PATCH}
  echo "Applying patch to OpenBLAS recipe in Elemental source for chooing a newer version"
fi

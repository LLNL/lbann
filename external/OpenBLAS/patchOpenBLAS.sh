#!/bin/sh

thisCommand=`basename $0`
PATCH_DIR=`dirname $0`
OpenBLAS_PATCH=${PATCH_DIR}/OpenBLAS.patch

# pick one file to change
fOrg=Makefile.power

# check if it contains the line to change
WRONG_OPT=`grep malign-power ${fOrg}`

if [ "${WRONG_OPT}" != "" ] ; then
  patch -p0 < ${OpenBLAS_PATCH}
  echo "Applying patch to OpenBLAS source"
fi

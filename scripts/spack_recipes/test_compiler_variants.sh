#!/bin/sh

DIRNAME=`dirname $0`
#Set Script Name variable
SCRIPT=`basename ${0}`

# Test normal optimized builds

#COMPILERS="gcc@7.1.0 gcc@4.9.3"
COMPILERS="gcc@7.1.0"
MPI_LIBS="mvapich2@2.2"

for c in ${COMPILERS}; do
  for m in ${MPI_LIBS}; do
    CMD="${DIRNAME}/build_lbann.sh -c ${c} -m ${m}"
    echo $CMD
    $CMD
    echo `pwd`
  done
done

# Now go into each directory and test the build
for f in *; do
    if [[ -d $f ]]; then
        # $f is a directory
        echo "Building LBANN in $f"
        cd $f/build
        make -j all
    fi
done

#!/bin/sh

SUCCESS=0
FAILURE=1

COMPILER_EXIT_CODE=${SUCCESS}
COMPILER_FAIL_COUNT=0

function print_results {
    RESULT=$?
    TEST=${1}
    if [ ${RESULT} -ne 0 ]; then
        COMPILER_EXIT_CODE=${FAILURE}
        ((COMPILER_FAIL_COUNT++))
        echo "***TEST******************************************************************************************************"
        echo "${TEST} FAILED"
        echo "*************************************************************************************************************"
    else
        echo "***TEST******************************************************************************************************"
        echo "${TEST} PASSED"
        echo "*************************************************************************************************************"
    fi
}


LBANN_DIR=$(git rev-parse --show-toplevel)
#Set Script Name variable
SCRIPT=`basename ${0}`

# Test normal optimized builds

COMPILERS="clang@4.0.0 gcc@4.9.3 gcc@7.1.0 intel@18.0.0"
MPI_LIBS="mvapich2@2.2"

for c in ${COMPILERS}; do
  for m in ${MPI_LIBS}; do
    CMD="${LBANN_DIR}/scripts/spack_recipes/build_lbann.sh -c ${c} -m ${m}"
    echo $CMD
    $CMD
    print_results "COMPILER: ${c} MPI_LIB: ${m}"
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
	print_results "${f} BUILD"
    fi
done

echo "***Fail Count: ${COMPILER_FAIL_COUNT}***"
exit ${COMPILER_EXIT_CODE}

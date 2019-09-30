#!/bin/bash -l

CLUSTER=$(hostname | sed 's/\([a-zA-Z][a-zA-Z]*\)[0-9]*/\1/g')

echo "allocate_and_run.sh CLUSTER="
echo $CLUSTER

WEEKLY=0
while :; do
    case ${1} in
        --weekly)
            # Run all tests. This is a weekly build.
            echo "Setting WEEKLY in allocate_and_run.sh"
            WEEKLY=1
            ;;
        -?*)
            # Unknown option
            echo "Unknown option (${1})" >&2
            exit 1
            ;;
        *)
            # Break loop if there are no more options
            break
    esac
    shift
done

echo "allocate_and_run.sh WEEKLY="
echo $WEEKLY

if [ "${CLUSTER}" = 'pascal' ]; then
    export MV2_USE_CUDA=1
fi

if [ "${CLUSTER}" = 'lassen' ]; then
    ALLOCATION_TIME_LIMIT=600
    if [ ${WEEKLY} -ne 0 ]; then
        timeout -k 5 24h bsub -G guests -Is -q pbatch -nnodes 16 -W $ALLOCATION_TIME_LIMIT ./run.sh --weekly
    else
        timeout -k 5 24h bsub -G guests -Is -q pbatch -nnodes 16 -W $ALLOCATION_TIME_LIMIT ./run.sh
    fi
elif [ "${CLUSTER}" = 'catalyst' ] || [ "${CLUSTER}" = 'corona' ] || [ "${CLUSTER}" = 'pascal' ]; then
    if [ ${WEEKLY} -ne 0 ]; then
        ALLOCATION_TIME_LIMIT=720
        timeout -k 5 24h salloc -N16 --partition=pbatch -t $ALLOCATION_TIME_LIMIT ./run.sh --weekly
        if [ "${CLUSTER}" = 'catalyst' ]; then
            cd integration_tests
            python -m pytest -s test_integration_performance_full_alexnet_clang6 --weekly --run --junitxml=../full_alexnet_clang6/results.xml
            python -m pytest -s test_integration_performance_full_alexnet_gcc7 --weekly --run --junitxml=../full_alexnet_gcc7/results.xml
            # python -m pytest -s test_integration_performance_full_alexnet_intel19 --weekly --run --junitxml=../full_alexnet_intel19/results.xml
            cd ..
        fi
    else
        if [ "${CLUSTER}" = 'catalyst' ]; then
            ALLOCATION_TIME_LIMIT=240
        elif [ "${CLUSTER}" = 'corona' ] || [ "${CLUSTER}" = 'pascal' ]; then
            ALLOCATION_TIME_LIMIT=660
        fi
        timeout -k 5 24h salloc -N16 --partition=pbatch -t $ALLOCATION_TIME_LIMIT ./run.sh
    fi
fi

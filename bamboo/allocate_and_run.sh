#!/bin/bash -l

CLUSTER=$(hostname | sed 's/\([a-zA-Z][a-zA-Z]*\)[0-9]*/\1/g')

echo "allocate_and_run.sh CLUSTER=${CLUSTER}"

export PYTHONPATH=${HOME}/.local/lib/python3.7/site-packages:${PYTHONPATH}

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

echo "allocate_and_run.sh WEEKLY=${WEEKLY}"

if [ "${CLUSTER}" = 'pascal' ]; then
    export MV2_USE_CUDA=1
fi

ALLOCATION_TIME_LIMIT_NIGHTLY=45
ALLOCATION_TIME_LIMIT_WEEKLY=90

if [ "${CLUSTER}" = 'lassen' ]; then
#    ALLOCATION_TIME_LIMIT=600
    if [ ${WEEKLY} -ne 0 ]; then
        timeout -k 5 24h bsub -G guests -Is -q pbatch -nnodes 4 -W ${ALLOCATION_TIME_LIMIT_WEEKLY} ./run.sh --weekly
    else
        timeout -k 5 24h bsub -G guests -Is -q pbatch -nnodes 2 -W ${ALLOCATION_TIME_LIMIT_NIGHTLY} ./run.sh
    fi
elif [ "${CLUSTER}" = 'ray' ]; then
#    ALLOCATION_TIME_LIMIT=240
    if [ ${WEEKLY} -ne 0 ]; then
        timeout -k 5 24h bsub -Is -q pbatch -nnodes 4 -W ${ALLOCATION_TIME_LIMIT_WEEKLY} ./run.sh --weekly
    else
        timeout -k 5 24h bsub -Is -q pbatch -nnodes 2 -W ${ALLOCATION_TIME_LIMIT_NIGHTLY} ./run.sh
    fi
elif [ "${CLUSTER}" = 'corona' ]; then
    if [ ${WEEKLY} -ne 0 ]; then
        ALLOCATION_TIME_LIMIT_WEEKLY=960
        timeout -k 5 24h salloc -N4 --partition=mi60 -t ${ALLOCATION_TIME_LIMIT_WEEKLY} ./run.sh --weekly
    else
        ALLOCATION_TIME_LIMIT_NIGHTLY=90 # Start with 1.5 hrs; may adjust for CPU clusters
        if [[ $(mjstat -c | awk 'match($1, "mi60") && NF < 7 { print $5 }') -ne "0" ]];
        then
            timeout -k 5 24h salloc -N2 --partition=mi60 -t ${ALLOCATION_TIME_LIMIT_NIGHTLY} ./run.sh
        else
            echo "Partition \"mi60\" on cluster \"${CLUSTER}\" appears to be down."
            echo "Trying \"mi25\"."
               timeout -k 5 24h salloc -N2 --partition=mi25 -t ${ALLOCATION_TIME_LIMIT_NIGHTLY} ./run.sh
        fi
    fi
elif [ "${CLUSTER}" = 'pascal' ]; then
#    ALLOCATION_TIME_LIMIT=960
    if [ ${WEEKLY} -ne 0 ]; then
        timeout -k 5 24h salloc -N4 --partition=pbatch -t ${ALLOCATION_TIME_LIMIT_WEEKLY} ./run.sh --weekly
    else
#        ALLOCATION_TIME_LIMIT=90 # Start with 1.5 hrs; may adjust for CPU clusters
        if [[ $(mjstat -c | awk 'match($1, "pbatch") && NF < 7 { print $5 }') -ne "0" ]];
        then
            timeout -k 5 24h salloc -N2 --partition=pvis -t ${ALLOCATION_TIME_LIMIT_NIGHTLY} ./run.sh
        else
            echo "Partition \"pbatch\" on cluster \"${CLUSTER}\" appears to be down."
        fi
    fi
elif [ "${CLUSTER}" = 'catalyst' ]; then
    if [ ${WEEKLY} -ne 0 ]; then
        ALLOCATION_TIME_LIMIT_WEEKLY=960
        timeout -k 5 24h salloc -N4 --partition=pbatch -t ${ALLOCATION_TIME_LIMIT_WEEKLY} ./run.sh --weekly
    else
        ALLOCATION_TIME_LIMIT_NIGHTLY=90 # Start with 1.5 hrs; may adjust for CPU clusters
        if [[ $(mjstat -c | awk 'match($1, "pbatch") && NF < 7 { print $5 }') -ne "0" ]];
        then
            timeout -k 5 24h salloc -N2 --partition=pbatch -t ${ALLOCATION_TIME_LIMIT_NIGHTLY} ./run.sh
        else
            echo "Partition \"pbatch\" on cluster \"${CLUSTER}\" appears to be down."
        fi
    fi
else
    echo "allocate_and_run.sh. Unsupported cluster CLUSTER=${CLUSTER}"
fi

CLUSTER=$(hostname | sed 's/\([a-zA-Z][a-zA-Z]*\)[0-9]*/\1/g')

if [ "${CLUSTER}" = 'catalyst' ]; then
    salloc -N16 -t 600 ./run.sh
fi

if [ "${CLUSTER}" = 'pascal' ]; then
    export MV2_USE_CUDA=1
    salloc -N16 -t 600 ./run.sh
fi

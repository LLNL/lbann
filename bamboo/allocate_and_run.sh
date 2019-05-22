CLUSTER=$(hostname | sed 's/\([a-zA-Z][a-zA-Z]*\)[0-9]*/\1/g')

if [ "${CLUSTER}" = 'catalyst' ]; then
    salloc -N16 -t 600 ./run.sh
fi

if [ "${CLUSTER}" = 'pascal' ]; then
    salloc -N16 -t 600 ./run.sh
fi

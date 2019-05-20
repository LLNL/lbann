CLUSTER=$(hostname | sed 's/\([a-zA-Z][a-zA-Z]*\)[0-9]*/\1/g')

if [ "${CLUSTER}" = 'catalyst' ]; then
    salloc -N16 -t 600 ./run.sh
fi

WEEKLY=0
while :; do
    case ${1} in
        --weekly)
            # Run all tests. This is a weekly build.
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

if [ "${CLUSTER}" = 'pascal' ]; then
    export MV2_USE_CUDA=1
    if [ ${WEEKLY} -ne 0 ]; then
        salloc -N16 -t 600 ./run.sh --weekly
    else
        salloc -N16 -t 600 ./run.sh
    fi

fi

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

if [ ${WEEKLY} -ne 0 ]; then
    salloc -N16 --partition=pbatch -t 900 ./run.sh --weekly
    if [ "${CLUSTER}" = 'catalyst' ]; then
        cd integration_tests
        python -m pytest -s test_integration_performance_full_alexnet_clang6 --weekly --run --junitxml=alexnet_clang6_results.xml
        python -m pytest -s test_integration_performance_full_alexnet_gcc7 --weekly --run --junitxml=alexnet_gcc7_results.xml
        # python -m pytest -s test_integration_performance_full_alexnet_intel19 --weekly --run --junitxml=alexnet_intel19_results.xml
        cd ..
    fi
else
    salloc -N16 --partition=pbatch -t 900 ./run.sh
fi

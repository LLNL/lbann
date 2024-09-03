# Run the sequential catch tests
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "~~~~~ Sequential catch tests"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

timeout -k 1m 10m \
        ${build_dir}/build-lbann/unit_test/seq-catch-tests \
        -r console \
        -r JUnit::out=${project_dir}/seq-tests_junit.xml \
    || {
    failed_tests=$(( ${failed_tests} + $? ))
    echo "******************************"
    echo " >>> seq-catch-tests FAILED"
    echo "******************************"
}

echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "~~~~~ MPI Catch Tests"
echo "-----   LBANN output logged to: ${project_dir}/lbann-log-mpi-catch-tests.log"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

case "${cluster}" in
    pascal)
        export OMPI_MCA_mpi_warn_on_fork=0
        timeout -k 1m 10m \
                srun -N1 -n2 --ntasks-per-node=2 --mpibind=off \
                -D ${build_dir}/build-lbann \
                ${build_dir}/build-lbann/unit_test/mpi-catch-tests \
                -r console::out=${project_dir}/mpi-catch-tests-console-rank=%r-size=%s.log \
                -r JUnit::out=${project_dir}/mpi-catch-tests-rank=%r-size=%s_junit.xml \
                > ${project_dir}/lbann-log-mpi-catch-tests.log 2>&1 \
            || {
            failed_tests=$((${failed_tests=} + $?))
            echo "******************************"
            echo " >>> mpi-catch-tests FAILED"
            echo "******************************"
        }
        ;;
    lassen)
        timeout -k 1m 10m \
                jsrun -n1 -r1 -a4 -c40 -g4 -d packed -b packed:10 \
                -h ${build_dir}/build-lbann \
                ${build_dir}/build-lbann/unit_test/mpi-catch-tests \
                -r console::out=${project_dir}/mpi-catch-tests-console-rank=%r-size=%s.log \
                -r JUnit::out=${project_dir}/mpi-catch-tests-rank=%r-size=%s_junit.xml \
                > ${project_dir}/lbann-log-mpi-catch-tests.log 2>&1 \
            || {
            failed_tests=$((${failed_tests} + $?))
            echo "******************************"
            echo " >>> mpi-catch-tests FAILED"
            echo "******************************"
        }
        ;;
    corona|tioga)
        export H2_SELECT_DEVICE_0=1
        timeout -k 1m 10m \
                flux run -N1 -n8 -g1 --exclusive \
                --cwd=${build_dir}/build-lbann \
                ${build_dir}/build-lbann/unit_test/mpi-catch-tests \
                -r console::out=${project_dir}/mpi-catch-tests-console-rank=%r-size=%s.log \
                -r JUnit::out=${project_dir}/mpi-catch-tests-rank=%r-size=%s_junit.xml \
                > ${project_dir}/lbann-log-mpi-catch-tests.log 2>&1 \
            || {
            failed_tests=$((${failed_tests} + $?))
            echo "******************************"
            echo " >>> mpi-catch-tests FAILED"
            echo "******************************"
        }
        ;;
    *)
        echo "Unknown cluster: ${cluster}"
        ;;
esac

for filename in ${project_dir}/mpi-catch-tests-console-rank=*.log; do
    [ -e "$filename" ] || continue
    echo "$filename"
    cat $filename
done

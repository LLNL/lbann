# Run the sequential catch tests
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "~~~~~ Sequential catch tests"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

timeout -k 1m 2m \
        ${build_dir}/build-lbann/unit_test/seq-catch-tests \
        -r console \
        -r JUnit::out=${project_dir}/seq-tests_junit.xml || {
    failed_tests=$(( ${failed_tests} + $? ))
    echo "******************************"
    echo " >>> SeqCatchTests FAILED"
    echo "******************************"
}

# if [[ -e "${build_dir}/build-lbann/bin/GPUCatchTests" ]]
# then
#     echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
#     echo "~~~~~ GPU tests"
#     echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

#     timeout -k 1m 2m \
#             ${build_dir}/build-lbann/bin/GPUCatchTests \
#             -r console \
#             -r JUnit::out=${project_dir}/gpu-tests_junit.xml || {
#         failed_tests=$(( ${failed_tests} + $? ))
#         echo "******************************"
#         echo " >>> GPUCatchTests FAILED"
#         echo "******************************"
#     }
# fi

echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "~~~~~ MPI tests"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

# FIXME (trb 20240702): There's an issue with needing each process to
# write out its own coverage database file, and I just need to sort
# through that before this will work. So for now, just run these when
# not running coverage tests.
if [[ -z "${run_coverage}" ]]
then
    case "${cluster}" in
        pascal)
            export OMPI_MCA_mpi_warn_on_fork=0
            timeout -k 1m 2m \
                    srun -N1 -n2 --ntasks-per-node=2 --mpibind=off \
                    -D ${build_dir}/build-lbann \
                    ${build_dir}/build-lbann/unit_test/mpi-catch-tests "exclude:[random]" "exclude:[filesystem]" \\
                    -r console::out=${project_dir}/mpi-tests-console-rank=%r-size=%s.log \
                    -r JUnit::out=${project_dir}/mpi-tests-rank=%r-size=%s_junit.xml || {
                failed_tests=$((${failed_tests=} + $?))
                echo "******************************"
                echo " >>> MPICatchTests FAILED"
                echo "******************************"
#                    -r mpicumulative \
            }
            timeout -k 1m 2m \
                    srun -N1 -n2 --ntasks-per-node=2 --mpibind=off \
                    -D ${build_dir}/build-lbann \
                    ${build_dir}/build-lbann/unit_test/mpi-catch-tests -s "[filesystem]" \
                    -r console::out=${project_dir}/mpi-catch-filesystem-tests-console-rank=%r-size=%s.log \
                    -r JUnit::out=${project_dir}/mpi-catch-filesystem-tests-rank=%r-size=%s_junit.xml || {
                failed_tests=$((${failed_tests} + $?))
                echo "******************************"
                echo " >>> mpi-catch-tests [filesystem] FAILED"
                echo "******************************"
                    # -r mpicumulative \
            }
            ;;
        lassen)
            timeout -k 1m 2m \
                    jsrun -n1 -r1 -a4 -c40 -g4 -d packed -b packed:10 \
                    ${build_dir}/build-lbann/unit_test/mpi-catch-tests "exclude:[random]" "exclude:[filesystem]" \
                    -r console::out=${project_dir}/mpi-tests-console-rank=%r-size=%s.log \
                    -r JUnit::out=${project_dir}/mpi-tests-rank=%r-size=%s_junit.xml || {
                failed_tests=$((${failed_tests} + $?))
                echo "******************************"
                echo " >>> mpi-catch-tests FAILED"
                echo "******************************"
#                    -r mpicumulative \
            }
            timeout -k 1m 2m \
                    jsrun -n1 -r1 -a4 -c40 -g4 -d packed -b packed:10 \
                    -h ${build_dir}/build-lbann \
                    ${build_dir}/build-lbann/unit_test/mpi-catch-tests -s "[filesystem]" \
                    -r console::out=${project_dir}/mpi-catch-filesystem-tests-console-rank=%r-size=%s.log \
                    -r JUnit::out=${project_dir}/mpi-catch-filesystem-tests-rank=%r-size=%s_junit.xml || {
                failed_tests=$((${failed_tests} + $?))
                echo "******************************"
                echo " >>> mpi-catch-tests [filesystem] FAILED"
                echo "******************************"
                    # -r mpicumulative \
            }
            ;;
        corona|tioga)
            export H2_SELECT_DEVICE_0=1
            timeout -k 1m 2m \
                    flux run -N1 -n8 -g1 --exclusive \
                    --cwd=${build_dir}/build-lbann \
                    ${build_dir}/build-lbann/unit_test/mpi-catch-tests "exclude:[random]" "exclude:[filesystem]" \
                    -r console::out=${project_dir}/mpi-tests-console-rank=%r-size=%s.log \
                    -r JUnit::out=${project_dir}/mpi-tests-rank=%r-size=%s_junit.xml || {
                failed_tests=$((${failed_tests} + $?))
                echo "******************************"
                echo " >>> mpi-catch-tests FAILED"
                echo "******************************"
                    # -r mpicumulative \
            }
# \
#                    -o "${project_dir}/mpi-catch-filesystem-results-rank=%r-size=%s.xml"
            timeout -k 1m 2m \
                    flux run -N1 -n8 -g1 --exclusive \
                    --cwd=${build_dir}/build-lbann \
                    ${build_dir}/build-lbann/unit_test/mpi-catch-tests -s "[filesystem]" \
                    -r console::out=${project_dir}/mpi-catch-filesystem-tests-console-rank=%r-size=%s.log \
                    -r JUnit::out=${project_dir}/mpi-catch-filesystem-tests-rank=%r-size=%s_junit.xml || {
                failed_tests=$((${failed_tests} + $?))
                echo "******************************"
                echo " >>> mpi-catch-tests [filesystem] FAILED"
                echo "******************************"
                    # -r mpicumulative \
            }
            ;;
        *)
            echo "Unknown cluster: ${cluster}"
            ;;
    esac
else
    echo "NOTE: Skipping MPI tests."
fi

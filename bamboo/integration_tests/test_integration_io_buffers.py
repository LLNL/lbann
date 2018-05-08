import sys
sys.path.insert(0, '../common_python')
import tools
import pytest
import os, sys
import common_code

def skeleton_io_buffers(cluster, dir_name, executables, compiler_name, weekly):
    if not weekly:
        pytest.skip('Not doing weekly testing')
    if cluster == 'surface':
        pytest.skip('skeleton_io_buffers does not run on surface')
    if compiler_name not in executables:
        pytest.skip('default_exes[%s] does not exist' % compiler_name)
    max_mb = 300
    # Printing output from 6*6*2=72 runs of LBANN makes the logs too slow.
    # Output from run_lbann is still printed - if there is a failure.
    should_log = False
    partitioned = 'mnist_partitioned_io'
    distributed = 'mnist_distributed_io'
    model_names = [partitioned, distributed]
    accuracies = {}
    errors = []
    all_values = []
    fatal_errors = []
    overall_min_partitioned_accuracy = float('inf')
    overall_min_distributed_accuracy = float('inf')
    for mini_batch_size in [300, 150, 100, 75, 60, 50]:
        num_models = max_mb / mini_batch_size
        for procs_per_model in [1, 2, 3, 4, 5, 6]:
            num_ranks = procs_per_model * num_models
            for model_name in model_names:
                output_file_name = '%s/bamboo/integration_tests/output/%s_%d_%d_output.txt' % (dir_name, model_name, mini_batch_size, procs_per_model)
                error_file_name = '%s/bamboo/integration_tests/error/%s_%d_%d_error.txt' % (dir_name, model_name, mini_batch_size, procs_per_model)
                command = tools.get_command(
                    cluster=cluster, executable=executables[compiler_name], num_nodes=2,
                    num_processes=num_ranks, dir_name=dir_name,
                    data_filedir_ray='/p/gscratchr/brainusr/datasets/MNIST',
                    data_reader_name='mnist', mini_batch_size=mini_batch_size,
                    model_folder='tests', model_name=model_name, num_epochs=5,
                    optimizer_name='adagrad',
                    processes_per_model=procs_per_model,
                    output_file_name=output_file_name, error_file_name=error_file_name)
                try:
                    common_code.run_lbann(command, model_name, output_file_name, error_file_name, should_log) # Don't need return value
                    accuracy_dict = common_code.extract_data(output_file_name, ['test_accuracy'], should_log)
                    accuracies[model_name] = accuracy_dict['test_accuracy']
                except Exception:
                    # We want to keep running to see if any other mini_batch_size & procs_per_model combination crashes.
                    # However, it is now pointless to compare accuracies.
                    fatal_errors.append('Crashed running %s with mini_batch_size=%d, procs_per_model=%d' % (model_name, mini_batch_size, procs_per_model))
            # End model name loop
            if fatal_errors == []:
                partitioned_num_models = len(accuracies[partitioned].keys())
                distributed_num_models = len(accuracies[distributed].keys())
                assert partitioned_num_models == distributed_num_models

                min_partitioned_accuracy = float('inf')
                min_distributed_accuracy = float('inf')
                for model_num in sorted(accuracies[partitioned].keys()):
                    partitioned_accuracy = accuracies[partitioned][model_num]['overall']
                    distributed_accuracy = accuracies[distributed][model_num]['overall']
                    if partitioned_accuracy < min_partitioned_accuracy:
                        min_partitioned_accuracy = partitioned_accuracy
                    if distributed_accuracy < min_distributed_accuracy:
                        min_distributed_accuracy = distributed_accuracy
                    tolerance = 0.05
                    # Are we within tolerance * expected_value?
                    if abs(partitioned_accuracy - distributed_accuracy) > abs(tolerance * min(partitioned_accuracy, distributed_accuracy)):
                        errors.append('partitioned = %f != %f = distributed; model_num=%s mini_batch_size=%d procs_per_model=%d' % (partitioned_accuracy, distributed_accuracy, model_num, mini_batch_size, procs_per_model))
                        all_values.append('partitioned = %f, %f = distributed; model_num=%s mini_batch_size=%d procs_per_model=%d' % (partitioned_accuracy, distributed_accuracy, model_num, mini_batch_size, procs_per_model))
                # End model_num loop
                if min_partitioned_accuracy < overall_min_partitioned_accuracy:
                    overall_min_partitioned_accuracy = min_partitioned_accuracy
                if min_distributed_accuracy < overall_min_distributed_accuracy:
                    overall_min_distributed_accuracy = min_distributed_accuracy
            # End fatal_errors == [] block
        # End procs_per_model loop
    # End mini_batch_size loop
    for fatal_error in fatal_errors:
        print(fatal_error)
    assert fatal_errors == []
    # If there were no fatal errors, archive the accuracies.
    if os.environ['LOGNAME'] == 'lbannusr':
        key = 'bamboo_planKey'
        if key in os.environ:
            plan = os.environ[key]
            if plan in ['LBANN-NIGHTD', 'LBANN-WD']:
                archive_file = '/usr/workspace/wsb/lbannusr/archives/%s/%s/%s/io_buffers.txt' % (plan, cluster, compiler_name)
                with open(archive_file, 'a') as archive:
                    archive.write('%s, %f, %f\n' % (os.environ['bamboo_buildNumber'], overall_min_partitioned_accuracy, overall_min_distributed_accuracy))
            else:
                print('The plan %s does not have archiving activated' % plan)
        else:
            print('%s is not in os.environ' % key)
    else:
        print('os.environ["LOGNAME"]=%s' % os.environ['LOGNAME'])

    print('Errors for: partitioned_and_distributed (%d)' % len(errors))
    for error in errors:
        print(error)
    if should_log:
        print('All values: (%d)' % len(all_values))
        for value in all_values:
            print(value)
    assert errors == []

def test_integration_io_buffers_clang4(cluster, dirname, exes, weekly):
    skeleton_io_buffers(cluster, dirname, exes, 'clang4', weekly)

def test_integration_io_buffers_gcc4(cluster, dirname, exes, weekly):
    skeleton_io_buffers(cluster, dirname, exes, 'gcc4', weekly)

def test_integration_io_buffers_gcc7(cluster, dirname, exes, weekly):
    skeleton_io_buffers(cluster, dirname, exes, 'gcc7', weekly)

def test_integration_io_buffers_intel18(cluster, dirname, exes, weekly):
    skeleton_io_buffers(cluster, dirname, exes, 'intel18', weekly)

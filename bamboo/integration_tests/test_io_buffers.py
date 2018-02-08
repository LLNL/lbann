import os, sys, pytest
import common_code

def test_partitioned_and_distributed_io_mnist(dirname, exe):
    slurm_cmd = 'srun'
    if os.getenv('SLURM_NNODES') is None:
        slurm_cmd = 'salloc -N2 ' + slurm_cmd
    max_mb = 300
    # Printing output from 6*6*2=72 runs of LBANN makes the logs too slow.
    # Output from run_lbann is still printed if there is a failure.
    should_log = False
    partitioned = 'mnist_partitioned_io'
    distributed = 'mnist_distributed_io'
    model_names = [partitioned, distributed]
    accuracies = {}
    errors = []
    all_values = []
    for mini_batch_size in [300, 150, 100, 75, 60, 50]:
        for procs_per_model in [1, 2, 3, 4, 5, 6]:
            num_ranks = procs_per_model * max_mb / mini_batch_size;
            for model_name in model_names:
                output_file_name = '%s/bamboo/integration_tests/%s_%d_%d.txt' % (dirname, model_name, mini_batch_size, procs_per_model)
                command = '%s -n%d %s --model=%s/model_zoo/tests/model_%s.prototext --optimizer=%s/model_zoo/optimizers/opt_adagrad.prototext --reader=%s/model_zoo/data_readers/data_reader_mnist.prototext --mini_batch_size=%d --num_epochs=5 --procs_per_model=%d > %s' % (slurm_cmd, num_ranks, exe, dirname, model_name, dirname, dirname, mini_batch_size, procs_per_model, output_file_name)
                common_code.run_lbann(command, model_name, output_file_name, should_log)
                accuracy_dict = common_code.extract_data(output_file_name, ['test_accuracy'], should_log)
                accuracies[model_name] = accuracy_dict['test_accuracy']
            
            partitioned_num_models = len(accuracies[partitioned].keys())
            distributed_num_models = len(accuracies[distributed].keys())
            assert partitioned_num_models == distributed_num_models
            
            for model_num in sorted(accuracies[partitioned].keys()):
                partitioned_accuracy = accuracies[partitioned][model_num]['overall']
                distributed_accuracy = accuracies[distributed][model_num]['overall']
                if abs(partitioned_accuracy - distributed_accuracy) > 0.01:
                    errors.append('partitioned = %d != %d = distributed; model_num=%s mini_batch_size=%d procs_per_model=%d' % (partitioned_accuracy, distributed_accuracy, model_num, mini_batch_size, procs_per_model))
                all_values.append('partitioned = %d, %d = distributed; model_num=%s mini_batch_size=%d procs_per_model=%d' % (partitioned_accuracy, distributed_accuracy, model_num, mini_batch_size, procs_per_model))
    
    print('Errors: (%d)' % len(errors))
    for error in errors:
        print(error)
    if should_log:
        print('All values: (%d)' % len(all_values))
        for value in all_values:
            print(value)
    assert errors == []

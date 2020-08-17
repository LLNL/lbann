import functools 
import operator 
import os 
import os.path 
import re
import sys
import pytest
import lbann 

current_file = os.path.realpath(__file__)
current_dir = os.path.dirname(current_file)
root_dir = os.path.dirname(current_dir)

import data.PROTEINS
import Sparse_Graph_Trainer

graph_dir = os.path.dirname(root_dir)
applications_dir = os.path.dirname(graph_dir)
lbann_dir = os.path.dirname(applications_dir)
common_python_dir = os.path.join(lbann_dir, 'bamboo/common_python')# Added lbann/bamboo/common_python 
sys.path.append(common_python_dir)
import tools


num_epochs = 100
mini_batch_size = 64
num_nodes = 2


expected_accuracy_range = (58, 75)

expected_mini_batch_times = {
       'ray' : 0.04
       }
expected_gpu_usage = {
        'ray' : 0.554
        }

def setup_experiment(lbann):
    """Construct LBANN experiment. 

    args: 
        lbann (module): Module for LBANN Python frontend
        
    """

    
    trainer = lbann.Trainer(mini_batch_size=mini_batch_size)



    callbacks = [lbann.CallbackPrint(), lbann.CallbackTimer(), lbann.CallbackGPUMemoryUsage()]
    

    
    model = Sparse_Graph_Trainer.make_model(dataset = 'PROTEINS',
                                            kernel_type = 'GCN',
                                            num_epochs = num_epochs,
                                            callbacks = callbacks)
    reader = data.PROTEINS.make_data_reader()
    
    # No validation set

    optimizer = lbann.Adam(learn_rate=0.01, beta1=0.9, beta2=0.99, eps=1e-8 )
    return trainer, model, reader, optimizer

# ==============================================
# Setup PyTest
# ==============================================

def augment_test_func(test_func):
    """Augment test function to parse log files.

    `tools.create_tests` creates functions that run an LBANN
    experiment. This function creates augmented functions that parse
    the log files after LBANN finishes running, e.g. to check metrics
    or runtimes.

    Note: The naive approach is to define the augmented test functions
    in a loop. However, Python closures are late binding. In other
    words, the function would be overwritten every time we define it.
    We get around this overwriting problem by defining the augmented
    function in the local scope of another function.

    Args:
        test_func (function): Test function created by
            `tools.create_tests`.

    Returns:
        function: Test that can interact with PyTest.

    """
    test_name = test_func.__name__

    # Define test function
    def func(cluster, exes, dirname):
        # Run LBANN experiment
        experiment_output = test_func(cluster, exes, dirname)

        # Parse LBANN log file
        train_accuracy = None
        gpu_usage = None
        mini_batch_times = []
        gpu_usages = []

        with open(experiment_output['stdout_log_file']) as f:
            for line in f:
                match = re.search('training epoch [0-9]+ accuracy : ([0-9.]+)%', line)
                if match:
                    train_accuracy = float(match.group(1))
                match = re.search('training epoch [0-9]+ mini-batch time statistics : ([0-9.]+)s mean', line)
                if match:
                    mini_batch_times.append(float(match.group(1)))
                match = re.search('GPU memory usage statistics : ([0-9.]+) GiB mean', line)
                if match:
                    gpu_usages.append(float(match.group(1)))
                    
        # Check if training accuracy is within expected range
        assert (expected_accuracy_range[0]
                < train_accuracy
                <expected_accuracy_range[1]), \
                'train accuracy is outside expected range'
       
        #Only tested on Ray. Skip if mini-batch test on another cluster. Change this when mini-batch values are available for other clusters 

        if (cluster == 'ray'):
        # Check if mini-batch time is within expected range
        # Note: Skip first epoch since its runtime is usually an outlier
            mini_batch_times = mini_batch_times[1:]
            mini_batch_time = sum(mini_batch_times) / len(mini_batch_times)
            assert (0.75 * expected_mini_batch_times[cluster]
                    < mini_batch_time
                    < 1.25 * expected_mini_batch_times[cluster]), \
                    'average mini-batch time is outside expected range'
        # Check for GPU usage and memory leaks 
        # Note: Skip first epoch 
            gpu_usages = gpu_usages[1:] 
            gpu_usage = sum(gpu_usages)/len(gpu_usages)

            assert (0.75 * expected_gpu_usage[cluster] 
                    < gpu_usage 
                    < 1.25 * expected_gpu_usage[cluster]),\
                    'average gpu usage is outside expected range'
    # Return test function from factory function
    func.__name__ = test_name
    return func

# Create test functions that can interact with PyTest
for _test_func in tools.create_tests(setup_experiment,
                                     __file__,
                                     nodes=num_nodes):
    globals()[_test_func.__name__] = augment_test_func(_test_func)


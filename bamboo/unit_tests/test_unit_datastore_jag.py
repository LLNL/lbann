import os.path
import re
import sys
import math
import numpy as np
import pytest

#
# The model below replicates:
#   model_zoo/tests/data_reader_tests/jag_single_layer_ae.prototext
#

# Local files
current_file = os.path.realpath(__file__)
current_dir = os.path.dirname(current_file)
sys.path.insert(0, os.path.join(os.path.dirname(current_dir), 'common_python'))

import tools

# ==============================================
# Options
# ==============================================

# Training options
num_epochs = 5
mini_batch_size = 128
num_nodes = 2
fc_neurons = 1024
jag_fraction = 0.01 # Train with 1000 out of 10000 samples
index_list = 't0_sample_list_10k.txt'
meta_data='--metadata=/usr/workspace/wsb/hysom/lbann/model_zoo/models/jag/wae_cycle_gan/jag_100M_metadata.prototext'
random_seed = 20191206

# ==============================================
# Setup LBANN experiment
# ==============================================

def setup_experiment(lbann):
    """Construct LBANN experiment.

    Args:
        lbann (module): Module for LBANN Python frontend

    """
    trainer = lbann.Trainer()
    model = construct_model(lbann)
    data_reader = construct_data_reader(lbann)
    optimizer = lbann.Adam(learn_rate=0.001, beta1 = 0.9, beta2 = 0.99, eps = 1e-08)

    return trainer, model, data_reader, optimizer

def construct_model(lbann):
    """Construct LBANN model.

    Args:
        lbann (module): Module for LBANN Python frontend
    """
    # TODO (tym): Figure out how to switch between LBANN builds. See
    # GitHub Issue #1289.
    import lbann.models

    # Layer graph
    input_ = lbann.Input(name='input', target_mode = 'N/A', io_buffer = 'partitioned')
    slice = lbann.Slice(input_, get_slice_points_from_reader='independent', name="slice_layer")
    x1 = lbann.Identity(slice, name='image_data_dummy')
    x2 = lbann.Identity(slice, name='param_data_id')
    f1 = lbann.FullyConnected(x1, num_neurons = fc_neurons, has_bias = True, name='encodefc1')
    f2 = lbann.FullyConnected(f1, hint_layer = 'image_data_dummy', has_bias = True, name='decode0')
    m1 = lbann.MeanSquaredError(parents = [f2, x1], name='img_loss')

    # Make sure all layers are on CPU
    for layer in lbann.traverse_layer_graph(input_):
        layer.device = 'cpu'

    # Objects for LBANN model
    callbacks = [lbann.CallbackPrint(interval = 1), lbann.CallbackTimer()]
    metrics = [lbann.Metric(m1, name='reconstr_loss')]
    obj = [m1, lbann.L2WeightRegularization(scale=1e-4)]

    # Construct model
    model = lbann.Model(mini_batch_size,
                       num_epochs,
                       layers=lbann.traverse_layer_graph(input_),
                       metrics=metrics,
                       objective_function = obj,
                       callbacks=callbacks,
                       random_seed=random_seed)
    pb_model = model.get_protobuf_model()
    pb_model.serialize_io = True
    return model

def construct_data_reader(lbann):
    """Construct Protobuf message for jag conduit data reader.

    Args:
        lbann (module): Module for LBANN Python frontend

    """
    # TODO (tym): Figure out how to switch between LBANN builds. See
    # GitHub Issue #1289.
    import lbann.contrib.lc.paths

    # Construct data reader
    message = lbann.reader_pb2.DataReader()
    message.requires_data_set_metadata = True
    reader = message.reader.add()

    # Configure data reader
    reader.name = 'jag_conduit'
    reader.role = 'train'
    reader.data_filedir = lbann.contrib.lc.paths.jag_dir()
    reader.index_list = index_list
    reader.percent_of_data_to_use = jag_fraction
    reader.num_labels = 5
    reader.disable_labels = True
    reader.disable_responses = True
    reader.shuffle = True

    return message

# ==============================================
# Setup PyTest
# ==============================================
def run_datastore_test_func(test_func, baseline_metrics, cluster, exes, dirname, profile_data) :
    '''Executes the input test function  

    Args:
        run_datastore_test_func (function): test function 
        baseline_metrics: list of metrics against which the output of
                          the test function will be compared
        profile_data: dictionary of key, value pairs for testing
                      entries in the output file: data_store_profile_train.txt

    Returns:
        list containg test name, pass/fail, etc.
        On error, this will have the form:
          ['FAILED', <test function name>, <error>]
        on success:
          ['passed', <test function name>]
    '''
    datastore_test_output = test_func(cluster, exes, dirname)

    test_name = test_func.__name__
    r = ['passed', test_name]
    datastore_metrics = []
    with open(datastore_test_output['stdout_log_file']) as f:
        for line in f:
            match = re.search('reconstr_loss : ([0-9.]+)', line)
            if match:
                datastore_metrics.append(float(match.group(1)))

    # Check if metrics are same in baseline and data store experiments
    # Note: "Print statistics" callback will print up to 6 digits
    # of metric values.
    if len(baseline_metrics) != len(datastore_metrics) :
        r[0] = 'FAILED'
        r.append('baseline and data store experiments did not run for same number of epochs; num baseline: ' + str(len(baseline_metrics)) + '; num ds: ' + str(len(datastore_metrics)))

    for i in range(len(datastore_metrics)):
        x = baseline_metrics[i]
        xhat = datastore_metrics[i]
        eps = np.finfo(np.float32).eps
        ceillogx = int(math.ceil(math.log10(x)))
        if abs(x-xhat) >= max(8*eps*x, 1.5*10**(ceillogx-6)) :
            r[0] = 'FAILED'
            r.append('found large discrepancy in metrics for baseline and data store experiments')

    # Check if entries profile_data exist and have correct values
    d = None
    for key in profile_data.keys() :
      if test_name.find(key) != -1 :
        d = profile_data[key]
        break
    assert d != None, 'failed to find key for profile_data'    

    found_profile_data = {}
    with open(datastore_test_output['work_dir'] + '/data_store_profile_train.txt') as f:
        for line in f:
            for key in d :
                if key in line and key not in found_profile_data.keys() :
                    t = line.split()
                    found_profile_data[key] = t[-1]

    for key in d.keys() :
        if key not in found_profile_data.keys() :
            r[0] = 'FAILED'
            r.append('missing key in profile_data: ' + key)
        elif found_profile_data[key] != d[key] :
            r[0] = 'FAILED'
            r.append('bad value for "' + key + '; value is: ' + str(found_profile_data[key]) + '; should be: ' + str(d[key]))
    return r

def run_baseline_test_func(baseline_test_func, cluster, exes, dirname) :
    '''Executes the input test function

    Args:
        baseline_test_func (function): test function 

    Returns:
        list of metrics that are parsed from the function's
        output log          
    '''
    baseline_test_output = baseline_test_func(cluster, exes, dirname)
    baseline_metrics = []
    with open(baseline_test_output['stdout_log_file']) as f:
        for line in f:
            match = re.search('reconstr_loss : ([0-9.]+)', line)
            if match:
                baseline_metrics.append(float(match.group(1)))
    
    assert len(baseline_metrics) > 0, 'failed to parse baseline_metrics; len: ' + str(len(baseline_metrics))
    return baseline_metrics

def create_test_func(baseline_test_func, datastore_test_funcs, profile_data=None) :
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
    # Define test function
    def func(cluster, exes, dirname, weekly):
        # Run LBANN experiment without data store
        baseline_metrics = run_baseline_test_func(baseline_test_func, cluster, exes, dirname)

        # Run LBANN experiments with data store
        num_failed = 0
        results = []
        for i in range(len(datastore_test_funcs)) :
            r = run_datastore_test_func(datastore_test_funcs[i], baseline_metrics, cluster, exes, dirname, profile_data)
            results.append(r)
            if len(r) > 2 :
              num_failed += 1

        work = []
        for x in results :
            work.append(' :: '.join(x))
        result_string = '\n'.join(work)
        assert num_failed == 0, '\n' + result_string

        print('\n===============================================')
        print('data_store test synopsis:')
        print(result_string)
        print('===============================================\n')

    # Return test function from factory function
    func.__name__ = baseline_test_func.__name__
    return func
 
# Create test functions that can interact with PyTest
def make_test(name, test_by_platform_list=[], args=[]) :
    test_list = tools.create_tests(
            setup_experiment,
            __file__,
            nodes=num_nodes,
            test_name_base=name,
            lbann_args=args)

    if test_by_platform_list != [] :
        for i in range(len(test_list)) :
            test_by_platform_list[i].append(test_list[i])
    return test_list

baseline_tests = make_test('nodatastore', args = [meta_data])

datastore_tests = [[] for j in range(len(baseline_tests))]

# Dictionary of dictionaries; this will contain data for testing
# the output file: data_store_profile_train.txt
profile_data = {}

# handles for entries in the profile_data dictionaries
is_e = 'is_explicitly_loading'
is_l = 'is_local_cache'
is_f = 'is_fully_loaded'

# test checkpoint, preload
test_name = 'data_store_checkpoint_preload'
make_test(test_name, datastore_tests, ['--preload_data_store', '--data_store_test_checkpoint=CHECKPOINT', '--data_store_profile', meta_data])
profile_data[test_name] =  {is_e : '0', is_l : '0', is_f : '1'}

# test checkpoint, explicit
test_name = 'data_store_checkpoint_explicit'
make_test(test_name, datastore_tests, ['--use_data_store', '--data_store_test_checkpoint=CHECKPOINT', '--data_store_profile', meta_data])
profile_data[test_name] =  {is_e : '1', is_l : '0', is_f : '0'}

# explicit loading
test_name = 'data_store_explicit'
make_test(test_name, datastore_tests, ['--use_data_store', '--data_store_profile', meta_data])
profile_data[test_name] = {is_e : '1', is_l : '0', is_f : '0'} 

# preloading
test_name = 'data_store_preload'
make_test(test_name, datastore_tests, ['--preload_data_store', '--data_store_profile', meta_data])
profile_data[test_name] =  {is_e : '0', is_l : '0', is_f : '1'}

for i in range(len(datastore_tests)):
    _test_func = create_test_func(baseline_tests[i], datastore_tests[i], profile_data)
    globals()[_test_func.__name__] = _test_func

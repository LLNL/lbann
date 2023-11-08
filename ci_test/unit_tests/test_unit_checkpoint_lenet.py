import os.path
import re
import sys
import math
import numpy as np
import google.protobuf.text_format
import pytest
import glob

# Local files
current_file = os.path.realpath(__file__)
current_dir = os.path.dirname(current_file)
sys.path.insert(0, os.path.join(os.path.dirname(current_dir), 'common_python'))
import tools

# ==============================================
# Options
# ==============================================

# Training options
num_epochs = 4
num_ckpt_epochs = int(float(num_epochs)/2)
mini_batch_size = 64
num_nodes = 1
lenet_fraction = 0.1
random_seed = 20191206

test_name_base='test_unit_checkpoint_lenet'
checkpoint_dir='ckpt'

# ==============================================
# Setup LBANN experiment
# ==============================================

def setup_experiment(lbann, weekly):
    """Construct LBANN experiment.

    Args:
        lbann (module): Module for LBANN Python frontend

    """
    trainer = lbann.Trainer(mini_batch_size=mini_batch_size,
                            random_seed=random_seed)

    # Checkpoint after every epoch
    trainer.callbacks = [
        lbann.CallbackCheckpoint(
            checkpoint_dir=checkpoint_dir,
            checkpoint_epochs=1,
            checkpoint_steps=845
        )
    ]

    model = construct_model(lbann)
    data_reader = construct_data_reader(lbann)
    optimizer = lbann.SGD(learn_rate=0.01, momentum=0.9)
    return trainer, model, data_reader, optimizer, None # Don't request any specific number of nodes

def construct_model(lbann):
    """Construct LBANN model.

    Args:
        lbann (module): Module for LBANN Python frontend

    """

    # TODO (tym): Figure out how to switch between LBANN builds. See
    # GitHub Issue #1289.
    import lbann.models

    # Manually override the global count so that each model is named the same
    lbann.models.LeNet.global_count = 0
    lbann.Layer.global_count = 0
    # Layer graph
    images = lbann.Input(data_field='samples')
    labels = lbann.Input(data_field='labels')
    x = lbann.models.LeNet(10)(images)
    probs = lbann.Softmax(x)
    loss = lbann.CrossEntropy(probs, labels)
    acc = lbann.CategoricalAccuracy(probs, labels)

    # Make sure all layers are on CPU
    for layer in lbann.traverse_layer_graph([images, labels]):
        layer.device = 'cpu'

    # Objects for LBANN model
    callbacks = [lbann.CallbackPrint(), lbann.CallbackTimer()]
    metrics = [lbann.Metric(acc, name='accuracy', unit='%')]

    # Construct model
    return lbann.Model(num_epochs,
                       layers=lbann.traverse_layer_graph([images, labels]),
                       objective_function=loss,
                       metrics=metrics,
                       callbacks=callbacks)

def construct_data_reader(lbann):
    """Construct Protobuf message for Python data reader.

    The Python data reader will import the current Python file to
    access the sample access functions.

    Args:
        lbann (module): Module for LBANN Python frontend

    """

    # TODO (tym): Figure out how to switch between LBANN builds. See
    # GitHub Issue #1289.
    import lbann.contrib.lc.paths

    # Load data readers from prototext
    dirname = os.path.dirname
    lbann_dir = dirname(dirname(dirname(os.path.realpath(__file__))))
    pb_file = os.path.join(lbann_dir,
                           'model_zoo',
                           'data_readers',
                           'data_reader_mnist.prototext')
    message = lbann.lbann_pb2.LbannPB()
    with open(pb_file, 'r') as f:
        google.protobuf.text_format.Merge(f.read(), message)
    message = message.data_reader

    # Set location of MNIST data
    for reader in message.reader:
        reader.data_filedir = lbann.contrib.lc.paths.mnist_dir()
        reader.fraction_of_data_to_use = lenet_fraction


    # Validation set
    message.reader[0].validation_fraction = 0.1

    return message

# ==============================================
# Setup PyTest
# ==============================================

def create_test_func(test_func):
    """Augment test function to cascade multiple tests and parse results.

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
    def func(cluster, dirname, weekly):

        # Run LBANN experiment baseline
        print('\n################################################################################')
        print('Running baseline model')
        print('################################################################################\n')
        baseline_test_output = test_func(cluster, dirname, weekly)
        baseline_training_metrics = tools.collect_metrics_from_log_func(baseline_test_output['stdout_log_file'], 'training epoch [0-9]+ objective function')
        baseline_validation_metrics = tools.collect_metrics_from_log_func(baseline_test_output['stdout_log_file'], 'validation objective function')
        baseline_test_metrics = tools.collect_metrics_from_log_func(baseline_test_output['stdout_log_file'], 'test objective function')

        # Run LBANN model to checkpoint
        print('\n################################################################################')
        print('Running initial model to checkpoint')
        print('################################################################################\n')
        test_func_checkpoint = tools.create_tests(
            setup_experiment,
            __file__,
            test_name_base=test_name_base,
            nodes=num_nodes,
            work_subdir='checkpoint',
            lbann_args=['--disable_cuda' + ' --num_epochs='+str(num_ckpt_epochs)],
        )

        checkpoint_test_output = test_func_checkpoint[0](cluster, dirname, weekly)
        checkpoint_training_metrics = tools.collect_metrics_from_log_func(checkpoint_test_output['stdout_log_file'], 'training epoch [0-9]+ objective function')
        checkpoint_validation_metrics = tools.collect_metrics_from_log_func(checkpoint_test_output['stdout_log_file'], 'validation objective function')
        checkpoint_test_metrics = tools.collect_metrics_from_log_func(checkpoint_test_output['stdout_log_file'], 'test objective function')

        print('\n################################################################################')
        print('Running restarted model to completion')
        print('################################################################################\n')
        test_func_restart = tools.create_tests(
            setup_experiment,
            __file__,
            test_name_base=test_name_base,
            nodes=num_nodes,
            work_subdir='restart',
            lbann_args=['--disable_cuda'
                        + ' --restart_dir='
                        + os.path.join(checkpoint_test_output['work_dir'], checkpoint_dir)
                        + ' --num_epochs='+str(num_epochs)],
        )

        # Restart LBANN model and run to completion
        restart_test_output = test_func_restart[0](cluster, dirname, weekly)
        restart_training_metrics = tools.collect_metrics_from_log_func(restart_test_output['stdout_log_file'], 'training epoch [0-9]+ objective function')
        restart_validation_metrics = tools.collect_metrics_from_log_func(restart_test_output['stdout_log_file'], 'validation objective function')
        restart_test_metrics = tools.collect_metrics_from_log_func(restart_test_output['stdout_log_file'], 'test objective function')

        print('\n################################################################################')
        print('Comparing results of models')
        print('################################################################################\n')

        # Check if metrics are same in baseline and test experiments
        # Note: "Print statistics" callback will print up to 6 digits
        # of metric values.

        # Comparing training objective functions
        tools.compare_metrics(baseline_training_metrics, checkpoint_training_metrics + restart_training_metrics)
        # Comparing validation objective functions
        tools.compare_metrics(baseline_validation_metrics, checkpoint_validation_metrics + restart_validation_metrics)
        # Comparing test objective functions
        tools.compare_metrics(baseline_test_metrics, restart_test_metrics)

        baseline_ckpt=os.path.join(baseline_test_output['work_dir'], checkpoint_dir)
        checkpoint_ckpt=os.path.join(checkpoint_test_output['work_dir'], checkpoint_dir)
        restart_ckpt=os.path.join(restart_test_output['work_dir'], checkpoint_dir)

        err = 0
        err_dirs = ''
        fileList = glob.glob('{base}/trainer0/*'.format(base=baseline_ckpt))
        fileList, tmp_err, tmp_err_str = tools.multidir_diff(baseline_ckpt, restart_ckpt, fileList)
        err += tmp_err
        err_dirs += tmp_err_str
        fileList, tmp_err, tmp_err_str = tools.multidir_diff(baseline_ckpt, checkpoint_ckpt, fileList)
        err += tmp_err
        err_dirs += tmp_err_str

        err_msg = "\nUnmatched checkpoints:\n"
        for f in fileList:
            err_msg += f + "\n"
        assert len(fileList) == 0, \
            'Extra checkpoint data in baseline directory: ' + err_msg
        assert err == 0, err_dirs

    # Return test function from factory function
    func.__name__ = test_name
    return func

# Create test functions that can interact with PyTest
for _test_func in tools.create_tests(setup_experiment,
                                     __file__,
                                     test_name_base=test_name_base,
                                     nodes=num_nodes,
                                     work_subdir='baseline',
                                     lbann_args=['--disable_cuda']):
    globals()[_test_func.__name__] = create_test_func(_test_func)

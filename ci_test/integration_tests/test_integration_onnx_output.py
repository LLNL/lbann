import functools
import operator
import os
import os.path
import re
import sys
import numpy as np
import google.protobuf.text_format
import pytest

# Local files
current_file = os.path.realpath(__file__)
current_dir = os.path.dirname(current_file)
sys.path.insert(0, os.path.join(os.path.dirname(current_dir), 'common_python'))
onnx_model = current_dir + '/experiments/test_integration_onnx_output/lbann.onnx'
import tools
import data.mnist

try:
    import onnxruntime
except ModuleNotFoundError:
    pytest.skip("Skipping ONNX runtime test; onnxruntime not found.",
                allow_module_level=True)

# ==============================================
# Options
# ==============================================

# Training options
num_epochs = 5
mini_batch_size = 64
num_nodes = 2


# ==============================================
# Setup LBANN experiment
# ==============================================

def setup_experiment(lbann):
    """Construct LBANN experiment.

    Args:
        lbann (module): Module for LBANN Python frontend

    """
    # Skip test if ONNX option not enabled
    if not lbann.has_feature("ONNX"):
        pytest.skip("This test requires ONNX.")

    trainer = lbann.Trainer(mini_batch_size=mini_batch_size)
    model = construct_model(lbann)

    data_reader = data.mnist.make_data_reader(lbann)
    # No validation set
    data_reader.reader[0].validation_fraction = 0

    optimizer = lbann.SGD(learn_rate=0.01, momentum=0.9)
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
    images = lbann.Input(data_field='samples', name="samples")
    labels = lbann.Input(data_field='labels', name="labels")
    x = lbann.models.LeNet(10)(images)
    probs = lbann.Softmax(x)
    loss = lbann.CrossEntropy(probs, labels)
    acc = lbann.CategoricalAccuracy(probs, labels)

    # Objects for LBANN model
    callbacks = [lbann.CallbackPrint(),
                 lbann.CallbackTimer(),
                 lbann.CallbackExportOnnx(
                     debug_string_filename='debug_onnx.txt')]
    metrics = [lbann.Metric(acc, name='accuracy', unit='%')]

    # Construct model
    return lbann.Model(num_epochs,
                       layers=lbann.traverse_layer_graph([images, labels]),
                       objective_function=loss,
                       metrics=metrics,
                       callbacks=callbacks)

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
    def func(cluster, dirname):

        # Run LBANN experiment
        experiment_output = test_func(cluster, dirname)

        # Run ONNX model in OnnxRuntime
        input0_name = 'samples_0'
        input1_name = 'labels_0'
        session = onnxruntime.InferenceSession(onnx_model, None)
        outputs = session.run(None, {input0_name: np.zeros((1,1,28,28), dtype=np.float32), input1_name: np.zeros((1,10), dtype=np.float32)})

    # Return test function from factory function
    func.__name__ = test_name
    return func

# Create test functions that can interact with PyTest
for _test_func in tools.create_tests(setup_experiment,
                                     __file__,
                                     nodes=num_nodes):
    globals()[_test_func.__name__] = augment_test_func(_test_func)

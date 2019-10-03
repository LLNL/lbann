import functools
import operator
import os
import os.path
import sys
import numpy as np

# Local files
current_file = os.path.realpath(__file__)
current_dir = os.path.dirname(current_file)
sys.path.insert(0, os.path.join(os.path.dirname(current_dir), 'common_python'))
import tools

# ==============================================
# Objects for Python data reader
# ==============================================
# Note: The Python data reader imports this file and calls the
# functions below to ingest data. This is the only part of the script
# that should be executed when the script is imported, or else the
# Python data reader might misbehave.

# Data
dictionary_size = 7
embedding_size = 5
np.random.seed(4321)
embedding_array = np.random.normal(size=(dictionary_size,embedding_size))

# Sample access functions
def get_sample(index):
    np.random.seed(1234+index)
    return [np.random.randint(dictionary_size)]
def num_samples():
    return 41
def sample_dims():
    return (1,)

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
    optimizer = lbann.NoOptimizer()
    return trainer, model, data_reader, optimizer

def construct_model(lbann):
    """Construct LBANN model.

    Args:
        lbann (module): Module for LBANN Python frontend

    """

    # Construct weights for embeddings
    embedding_values = ' '.join([str(i) for i in np.nditer(embedding_array)])
    init = lbann.ValueInitializer(values=embedding_values)
    w = lbann.Weights(optimizer=lbann.SGD(), initializer=init)

    # Layer graph
    input = lbann.Input()
    embedding = lbann.Embedding(input,
                                weights=w,
                                dictionary_size=dictionary_size,
                                embedding_size=embedding_size,
                                device='cpu')
    l2_norm2 = lbann.L2Norm2(embedding)
    layers = list(lbann.traverse_layer_graph(input))
    metric = lbann.Metric(l2_norm2, name='L2 norm squared')
    obj = lbann.ObjectiveFunction(l2_norm2)

    # Compute expected value
    metric_vals = []
    for i in range(num_samples()):
        input = get_sample(i)
        embedding = embedding_array[int(input[0]), :]
        l2_norm2 = np.inner(embedding, embedding)
        metric_vals.append(l2_norm2)
    expected_metric_value = np.mean(metric_vals)
    tol = 8 * expected_metric_value * np.finfo(np.float32).eps

    # Initialize check metric callback
    callbacks = [lbann.CallbackCheckMetric(metric='L2 norm squared',
                                           lower_bound=expected_metric_value-tol,
                                           upper_bound=expected_metric_value+tol,
                                           error_on_failure=True,
                                           execution_modes='test'),
                 lbann.CallbackCheckGradients(error_on_failure=True)]

    # Construct model
    mini_batch_size = 17
    num_epochs = 0
    return lbann.Model(mini_batch_size,
                       num_epochs,
                       layers=layers,
                       objective_function=obj,
                       metrics=[metric],
                       callbacks=callbacks)

def construct_data_reader(lbann):
    """Construct Protobuf message for Python data reader.

    The Python data reader will import the current Python file to
    access the sample access functions.

    Args:
        lbann (module): Module for LBANN Python frontend

    """

    # Note: The training data reader should be removed when
    # https://github.com/LLNL/lbann/issues/1098 is resolved.
    message = lbann.reader_pb2.DataReader()
    message.reader.extend([
        tools.create_python_data_reader(
            lbann,
            current_file,
            'get_sample',
            'num_samples',
            'sample_dims',
            'train'
        )
    ])
    message.reader.extend([
        tools.create_python_data_reader(
            lbann,
            current_file,
            'get_sample',
            'num_samples',
            'sample_dims',
            'test'
        )
    ])
    return message

# ==============================================
# Setup PyTest
# ==============================================

# Create test functions that can interact with PyTest
# Note: Create test name by removing ".py" from file name
_test_name = os.path.splitext(os.path.basename(current_file))[0]
for test in tools.create_tests(setup_experiment, _test_name):
    globals()[test.__name__] = test

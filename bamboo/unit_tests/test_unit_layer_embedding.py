import functools
import operator
import os
import os.path
import sys
import numpy as np

# Bamboo utilities
current_file = os.path.realpath(__file__)
current_dir = os.path.dirname(current_file)
sys.path.insert(0, os.path.join(os.path.dirname(current_dir), 'common_python'))
import tools

# ==============================================
# Objects for Python data reader
# ==============================================
# Note: The Python data reader imports this file as a module and calls
# the functions below to ingest data.

# Data
_num_samples = 41
_num_embeddings = 7

# Sample access functions
def get_sample(index):
    np.random.seed(2019101500+index)
    i = np.random.randint(_num_embeddings)
    if index in (1,2,4,7,17,31):
        i = 0
    return [i]
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

    # Input data
    x = lbann.Identity(lbann.Input())
    x_lbann = x

    # Objects for LBANN model
    obj = []
    metrics = []
    callbacks = []

    # ------------------------------------------
    # No padding index
    # ------------------------------------------

    # Embeddings
    np.random.seed(20191015)
    embedding_dim = 5
    embeddings = np.random.normal(size=(_num_embeddings,embedding_dim))

    # LBANN implementation
    embedding_weights = lbann.Weights(
        optimizer=lbann.SGD(),
        initializer=lbann.ValueInitializer(values=tools.str_list(np.nditer(embeddings)))
    )
    x = x_lbann
    y = lbann.Embedding(x,
                        weights=embedding_weights,
                        num_embeddings=_num_embeddings,
                        embedding_dim=embedding_dim)
    z = lbann.L2Norm2(y)
    obj.append(z)
    metrics.append(lbann.Metric(z, name='no padding index'))

    # NumPy implementation
    vals = []
    for i in range(num_samples()):
        x = get_sample(i)[0]
        y = embeddings[x]
        z = tools.numpy_l2norm2(y)
        vals.append(z)
    val = np.mean(vals)
    tol = 8 * val * np.finfo(np.float32).eps
    callbacks.append(lbann.CallbackCheckMetric(
        metric=metrics[-1].name,
        lower_bound=val-tol,
        upper_bound=val+tol,
        error_on_failure=True,
        execution_modes='test'))

    # ------------------------------------------
    # Padding index 0
    # ------------------------------------------

    # Embeddings
    np.random.seed(201910152)
    embedding_dim = 3
    padding_idx = 0
    embeddings = np.random.normal(size=(_num_embeddings,embedding_dim))

    # LBANN implementation
    # Note: Embedding layer gradients are not exact if a padding index
    # is set. Avoid gradient checking by not using an optimizer.
    embedding_weights = lbann.Weights(
        optimizer=None,
        initializer=lbann.ValueInitializer(values=tools.str_list(np.nditer(embeddings)))
    )
    x = x_lbann
    y = lbann.Embedding(x,
                        weights=embedding_weights,
                        num_embeddings=_num_embeddings,
                        embedding_dim=embedding_dim,
                        padding_idx=padding_idx)
    z = lbann.L2Norm2(y)
    obj.append(z)
    metrics.append(lbann.Metric(z, name='padding index = 0'))

    # NumPy implementation
    vals = []
    for i in range(num_samples()):
        x = get_sample(i)[0]
        if x == padding_idx:
            y = np.zeros(shape=embedding_dim)
        else:
            y = embeddings[x]
        z = tools.numpy_l2norm2(y)
        vals.append(z)
    val = np.mean(vals)
    tol = 8 * val * np.finfo(np.float32).eps
    callbacks.append(lbann.CallbackCheckMetric(
        metric=metrics[-1].name,
        lower_bound=val-tol,
        upper_bound=val+tol,
        error_on_failure=True,
        execution_modes='test'))

    # ------------------------------------------
    # Gradient checking
    # ------------------------------------------

    callbacks.append(lbann.CallbackCheckGradients(error_on_failure=True))

    # ------------------------------------------
    # Construct model
    # ------------------------------------------

    # Construct model
    mini_batch_size = num_samples() // 2
    num_epochs = 0
    return lbann.Model(mini_batch_size,
                       num_epochs,
                       layers=lbann.traverse_layer_graph(x_lbann),
                       objective_function=obj,
                       metrics=metrics,
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
for test in tools.create_tests(setup_experiment, __file__):
    globals()[test.__name__] = test

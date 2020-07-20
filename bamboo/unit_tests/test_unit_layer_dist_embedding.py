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
_seed = 20200117
_num_samples = 41
_num_embeddings = 11
_sequence_length = 3

# Sample access functions
def get_sample(index):
    np.random.seed(100*_seed+index)
    return np.random.randint(_num_embeddings, size=_sequence_length)
def num_samples():
    return _num_samples
def sample_dims():
    return (_sequence_length,)

# ==============================================
# Setup LBANN experiment
# ==============================================

def setup_experiment(lbann):
    """Construct LBANN experiment.

    Args:
        lbann (module): Module for LBANN Python frontend

    """
    mini_batch_size = num_samples() // 2
    trainer = lbann.Trainer(mini_batch_size)
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
    # GPU
    # ------------------------------------------

    # Embeddings
    np.random.seed(_seed)
    embedding_dim = 7
    embeddings = np.random.normal(size=(_num_embeddings,embedding_dim))

    # LBANN implementation
    embedding_weights = lbann.Weights(
        optimizer=lbann.SGD(),
        initializer=lbann.ValueInitializer(values=tools.str_list(np.nditer(embeddings)))
    )
    x = x_lbann
    y = lbann.DistEmbedding(x,
                            weights=embedding_weights,
                            num_embeddings=_num_embeddings,
                            embedding_dim=embedding_dim,
                            barrier_in_forward_prop=True,
                            device='gpu')
    z = lbann.L2Norm2(y)
    obj.append(z)
    metrics.append(lbann.Metric(z, name='GPU'))

    # NumPy implementation
    vals = []
    for i in range(num_samples()):
        x = get_sample(i)
        y = embeddings[x,:]
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
    # CPU
    # ------------------------------------------

    # Embeddings
    np.random.seed(_seed)
    embedding_dim = 5
    embeddings = np.random.normal(size=(_num_embeddings,embedding_dim))

    # LBANN implementation
    embedding_weights = lbann.Weights(
        optimizer=lbann.SGD(),
        initializer=lbann.ValueInitializer(values=tools.str_list(np.nditer(embeddings)))
    )
    x = x_lbann
    y = lbann.DistEmbedding(x,
                            weights=embedding_weights,
                            num_embeddings=_num_embeddings,
                            embedding_dim=embedding_dim,
                            barrier_in_forward_prop=True,
                            device='cpu')
    z = lbann.L2Norm2(y)
    obj.append(z)
    metrics.append(lbann.Metric(z, name='CPU'))

    # NumPy implementation
    vals = []
    for i in range(num_samples()):
        x = get_sample(i)
        y = embeddings[x,:]
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
    num_epochs = 0
    return lbann.Model(num_epochs,
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
# Note (tym 6/12/20): Tests are disabled for now since the default
# build doesn't include SHMEM or NVSHMEM. Restore these tests when
# proper support is added.
# for test in tools.create_tests(setup_experiment, __file__):
#     globals()[test.__name__] = test

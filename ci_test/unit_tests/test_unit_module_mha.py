import os
import os.path
import sys
import numpy as np
import pytest

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
np.random.seed(20230510)
_num_samples = 2

_sequence_length = 32
embed_dim = 64
num_heads = 8

_samples = np.random.normal(size=(_num_samples, 3, _sequence_length,
                                  embed_dim)).astype(np.float32)

# ------------------------------------------
# PyTorch implementation
# ------------------------------------------
try:
    import torch
except (ModuleNotFoundError, ImportError):
    pytest.skip('PyTorch is required for this test', allow_module_level=True)

torch.manual_seed(20230510)
mha_module = torch.nn.MultiheadAttention(embed_dim=embed_dim,
                                         num_heads=num_heads,
                                         batch_first=True,
                                         bias=True)

# Weights are packed in the PyTorch module by default, unpack to q/k/v
q_w, k_w, v_w = mha_module.in_proj_weight.detach().split(embed_dim)
q_b, k_b, v_b = mha_module.in_proj_bias.detach().split(embed_dim)

# Compute reference values
with torch.no_grad():
    q, k, v = torch.from_numpy(_samples).unbind(dim=1)
    acts_pt, _ = mha_module(q, k, v)
    acts_np = acts_pt.detach().cpu().numpy()

# Store samples along with reference values
_samples_with_validation = np.concatenate(
    (_samples, acts_np.reshape(_num_samples, 1, _sequence_length, embed_dim)),
    axis=1)


# Sample access functions
def get_sample(index):
    return _samples_with_validation[index].flatten()


def num_samples():
    return _num_samples


def sample_dims():
    return (4 * _sequence_length, embed_dim)


# ==============================================
# Setup LBANN experiment
# ==============================================

err_tol = abs(8 * np.mean(_samples_with_validation[:, -1]) *
              np.finfo(np.float32).eps)


def setup_experiment(lbann, weekly):
    """Construct LBANN experiment.

    Args:
        lbann (module): Module for LBANN Python frontend

    """
    mini_batch_size = num_samples()
    trainer = lbann.Trainer(mini_batch_size)
    model = construct_model(lbann)
    data_reader = construct_data_reader(lbann)
    optimizer = lbann.NoOptimizer()
    return trainer, model, data_reader, optimizer, None  # Don't request any specific number of nodes


def construct_model(lbann):
    """Construct LBANN model.

    Args:
        lbann (module): Module for LBANN Python frontend

    """

    # ------------------------------------------
    # LBANN implementation
    # ------------------------------------------

    # Objects for LBANN model
    metrics = []
    callbacks = []

    # Input data
    x = lbann.Slice(
        lbann.Input(data_field='samples'),
        slice_points=[
            0, _sequence_length, _sequence_length * 2, _sequence_length * 3,
            _sequence_length * 4
        ],
    )

    q = lbann.Identity(x)
    k = lbann.Identity(x)
    v = lbann.Identity(x)
    verification = lbann.Identity(x)

    # Test module
    from lbann.modules.transformer import MultiheadAttention
    mha = MultiheadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
    )

    mha.query_weights = [
        lbann.Weights(initializer=lbann.ValueInitializer(
            values=np.nditer(q_w.transpose(0, 1).contiguous()))),
        lbann.Weights(initializer=lbann.ValueInitializer(
            values=np.nditer(q_b.contiguous()))),
    ]
    mha.key_weights = [
        lbann.Weights(initializer=lbann.ValueInitializer(
            values=np.nditer(k_w.transpose(0, 1).contiguous()))),
        lbann.Weights(initializer=lbann.ValueInitializer(
            values=np.nditer(k_b.contiguous()))),
    ]
    mha.value_weights = [
        lbann.Weights(initializer=lbann.ValueInitializer(
            values=np.nditer(v_w.transpose(0, 1).contiguous()))),
        lbann.Weights(initializer=lbann.ValueInitializer(
            values=np.nditer(v_b.contiguous()))),
    ]
    mha.output_weights = [
        lbann.Weights(initializer=lbann.ValueInitializer(
            values=np.nditer(mha_module.out_proj.weight.detach().cpu().
                             transpose(0, 1).contiguous().numpy()))),
        lbann.Weights(initializer=lbann.ValueInitializer(values=np.nditer(
            mha_module.out_proj.bias.detach().cpu().contiguous().numpy()))),
    ]

    # TODO: Test mask and self-attention
    acts = mha(q, k, v)

    # Compute MSE loss w.r.t. verification tensor
    l2_loss = lbann.MeanSquaredError(acts, verification)
    metrics.append(lbann.Metric(l2_loss, name='acts'))

    callbacks.append(
        lbann.CallbackCheckMetric(metric=metrics[-1].name,
                                  lower_bound=0,
                                  upper_bound=err_tol,
                                  error_on_failure=True,
                                  execution_modes='test'))

    # ------------------------------------------
    # Construct model
    # ------------------------------------------

    num_epochs = 0
    return lbann.Model(num_epochs,
                       layers=lbann.traverse_layer_graph(x),
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
        tools.create_python_data_reader(lbann, current_file, 'get_sample',
                                        'num_samples', 'sample_dims', 'train')
    ])
    message.reader.extend([
        tools.create_python_data_reader(lbann, current_file, 'get_sample',
                                        'num_samples', 'sample_dims', 'test')
    ])
    return message


# ==============================================
# Setup PyTest
# ==============================================

# Create test functions that can interact with PyTest
# Note: Create test name by removing ".py" from file name
_test_name = os.path.splitext(os.path.basename(current_file))[0]
for _test_func in tools.create_tests(setup_experiment, _test_name):
    globals()[_test_func.__name__] = _test_func

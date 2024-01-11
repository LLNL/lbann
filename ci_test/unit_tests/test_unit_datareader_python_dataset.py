import os
import os.path
import sys
import numpy as np
from lbann.util.data import Dataset, Sample, SampleDims

# Bamboo utilities
current_file = os.path.realpath(__file__)
current_dir = os.path.dirname(current_file)
sys.path.insert(0, os.path.join(os.path.dirname(current_dir), 'common_python'))
import tools

# ==============================================
# Objects for Python dataset data reader
# ==============================================
# Note: The Python dataset data reader loads the dataset constructed below.

# Data
class TestDataset(Dataset):
    def __init__(self):
        np.random.seed(20240109)
        self.num_samples = 29
        self.sample_size = 7
        self.samples = np.random.normal(size=(self.num_samples,self.sample_size)).astype(np.float32)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, index):
        return Sample(sample=self.samples[index,:])
    
    def get_sample_dims(self):
        return SampleDims(sample=[self.sample_size])

test_dataset = TestDataset()

# ==============================================
# Setup LBANN experiment
# ==============================================

def setup_experiment(lbann, weekly):
    """Construct LBANN experiment.

    Args:
        lbann (module): Module for LBANN Python frontend

    """
    mini_batch_size = len(test_dataset) // 4
    trainer = lbann.Trainer(mini_batch_size)
    model = construct_model(lbann)
    data_reader = construct_data_reader(lbann)
    optimizer = lbann.NoOptimizer()
    return trainer, model, data_reader, optimizer, None # Don't request any specific number of nodes

def construct_model(lbann):
    """Construct LBANN model.

    Args:
        lbann (module): Module for LBANN Python frontend

    """

    # Layer graph
    x = lbann.Input(data_field='samples')
    y = lbann.L2Norm2(x)
    layers = list(lbann.traverse_layer_graph(x))
    metric = lbann.Metric(y, name='obj')
    callbacks = []

    # Compute expected value with NumPy
    vals = []
    for i in range(len(test_dataset)):
        x = test_dataset[i].sample.astype(np.float64)
        y = tools.numpy_l2norm2(x)
        vals.append(y)
    val = np.mean(vals)
    tol = 8 * val * np.finfo(np.float32).eps
    callbacks.append(lbann.CallbackCheckMetric(
        metric=metric.name,
        lower_bound=val-tol,
        upper_bound=val+tol,
        error_on_failure=True,
        execution_modes='test'))

    # Construct model
    num_epochs = 0
    return lbann.Model(num_epochs,
                       layers=layers,
                       metrics=[metric],
                       callbacks=callbacks)

def construct_data_reader(lbann):
    """Construct Protobuf message for Python dataset data reader.

    The Python data reader will import the current Python file to
    access the sample access functions.

    Args:
        lbann (module): Module for LBANN Python frontend

    """

    # Note: The training data reader should be removed when
    # https://github.com/LLNL/lbann/issues/1098 is resolved.
    message = lbann.reader_pb2.DataReader()
    message.reader.extend([
        tools.create_python_dataset_reader(
            lbann,
            __file__,
            test_dataset,
            'train'
        )
    ])
    message.reader.extend([
        tools.create_python_dataset_reader(
            lbann,
            __file__,
            test_dataset,
            'test'
        )
    ])
    return message

# ==============================================
# Setup PyTest
# ==============================================

# Create test functions that can interact with PyTest
for _test_func in tools.create_tests(setup_experiment, __file__):
    globals()[_test_func.__name__] = _test_func

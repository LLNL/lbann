import copy
import os
import os.path
import sys
import numpy as np
from lbann.util.data import Dataset, Sample, SampleDims, construct_python_dataset_reader

# Bamboo utilities
current_file = os.path.realpath(__file__)
current_dir = os.path.dirname(current_file)
sys.path.insert(0, os.path.join(os.path.dirname(current_dir), 'common_python'))
import tools

# ==============================================
# Objects for Python dataset data reader
# ==============================================

_num_samples = 83


# Data
class TestDataset(Dataset):

    def __init__(self, record=False):
        np.random.seed(20240502)
        self.num_samples = _num_samples
        self.sample_size = 7
        self.samples = np.random.normal(size=(self.num_samples,
                                              self.sample_size)).astype(
                                                  np.float32)
        self.indices_read = []
        self.record = record

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        if index in self.indices_read:
            raise ValueError(f'Read index {index} twice')
        if self.record:
            with open(f'lbann_loaded_samples.{os.getpid()}', 'a') as fp:
                fp.write(f'{index}\n')

        self.indices_read.append(index)
        return Sample(sample=self.samples[index, :])

    def get_sample_dims(self):
        return SampleDims(sample=[self.sample_size])


# ==============================================
# Setup LBANN experiment
# ==============================================


def setup_experiment(lbann, weekly):
    """Construct LBANN experiment.

    Args:
        lbann (module): Module for LBANN Python frontend

    """
    # Remove files from last experiment, if exist
    for f in os.listdir(work_dir):
        if 'lbann_loaded_samples.' in f:
            os.unlink(os.path.join(work_dir, f))

    mini_batch_size = _num_samples // 4
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

    # Layer graph
    x = lbann.Input(data_field='samples')
    y = lbann.L2Norm2(x)
    layers = list(lbann.traverse_layer_graph(x))
    metric = lbann.Metric(y, name='obj')
    callbacks = []

    # Compute expected value with NumPy
    test_dataset = TestDataset()
    vals = []
    for i in range(len(test_dataset)):
        x = test_dataset[i].sample.astype(np.float64)
        y = tools.numpy_l2norm2(x)
        vals.append(y)

    val = np.mean(vals)
    tol = 8 * val * np.finfo(np.float32).eps
    callbacks.append(
        lbann.CallbackCheckMetric(metric=metric.name,
                                  lower_bound=val - tol,
                                  upper_bound=val + tol,
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

    dataset_path = os.path.join(work_dir, 'dataset.pkl')

    # Note: The training data reader should be removed when
    # https://github.com/LLNL/lbann/issues/1098 is resolved.
    message = lbann.reader_pb2.DataReader()
    message.reader.extend([
        construct_python_dataset_reader(TestDataset(record=False),
                                        dataset_path,
                                        'train',
                                        shuffle=True),
        construct_python_dataset_reader(TestDataset(record=True),
                                        dataset_path,
                                        'test',
                                        shuffle=True),
    ])
    return message


def post_test(lbann, weekly):
    # Check that all indices were read by data reader
    files = [f for f in os.listdir(work_dir) if 'lbann_loaded_samples.' in f]
    print(f'{len(files)} files found')
    indices = []
    for f in files:
        with open(os.path.join(work_dir, f), 'r') as fp:
            indices.extend([int(l.strip()) for l in fp.readlines()])
    print(f'Number of indices: {len(indices)}')
    if len(indices) != _num_samples:
        raise ValueError(f'Invalid number of samples (read: {len(indices)}, '
                         f'expected: {_num_samples})')
    if list(sorted(indices)) != list(range(_num_samples)):
        raise ValueError('Duplicate indices read by dataset reader')


# ==============================================
# Setup PyTest
# ==============================================

work_dir = os.path.join(os.path.dirname(__file__), 'experiments',
                        os.path.basename(__file__).split('.py')[0])
os.makedirs(work_dir, exist_ok=True)

# Create test functions that can interact with PyTest
for _test_func in tools.create_tests(setup_experiment,
                                     __file__,
                                     work_dir=work_dir,
                                     post_test_func=post_test):
    globals()[_test_func.__name__] = _test_func

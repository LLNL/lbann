import os
import os.path
import sys
import pytest
import numpy as np
import lbann.contrib.args
from lbann.util.data import DistConvDataset, Sample, SampleDims, construct_python_dataset_reader

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
class TestDataset(DistConvDataset):
    def __init__(self):
        np.random.seed(20240109)
        self.num_samples = 29
        self.sample_size = 8
        self.samples = np.random.normal(size=(self.num_samples,
                                              1,
                                              self.sample_size,
                                              self.sample_size,
                                              self.sample_size)).astype(np.float32)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, index):
        slice_ind = self.rank % self.num_io_partitions
        slice_width = self.sample_size // self.num_io_partitions
        return Sample(sample=np.ascontiguousarray(self.samples[index,:,slice_ind*slice_width:(slice_ind+1)*slice_width,:,:]))
    
    def get_sample_dims(self):
        return SampleDims(sample=[1, self.sample_size, self.sample_size, self.sample_size])

test_dataset = TestDataset()

# ==============================================
# Setup LBANN experiment
# ==============================================

def setup_experiment(lbann, weekly):
    """Construct LBANN experiment.

    Args:
        lbann (module): Module for LBANN Python frontend

    """
    if not lbann.has_feature('DISTCONV'):
        message = f'{os.path.basename(__file__)} requires DISTCONV'
        print('Skip - ' + message)
        pytest.skip(message)

    mini_batch_size = len(test_dataset) // 4
    trainer = lbann.Trainer(mini_batch_size)
    model = construct_model(lbann)
    data_reader = construct_data_reader(lbann)
    optimizer = lbann.NoOptimizer()
    return trainer, model, data_reader, optimizer, None # Don't request any specific number of nodes

def create_parallel_strategy(num_height_groups):
    return {"depth_groups": num_height_groups}

def construct_model(lbann):
    """Construct LBANN model.

    Args:
        lbann (module): Module for LBANN Python frontend

    """

    num_height_groups = tools.gpus_per_node(lbann)
    if num_height_groups == 0:
        e = 'this test requires GPUs.'
        print('Skip - ' + e)
        pytest.skip(e)

    # Layer graph
    ps = create_parallel_strategy(num_height_groups)
    x = lbann.Input(data_field='samples', parallel_strategy=ps)
    kernel = np.zeros([1,1,3,3,3])
    kernel[0,0,1,1,1] = 1
    kernel_weights = lbann.Weights(
        initializer=lbann.ValueInitializer(values=np.nditer(kernel)),
        name=f'kernel'
    )
    x = lbann.Convolution(
        x,
        weights=(kernel_weights,),
        num_dims=3,
        out_channels=1,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        has_bias=False,
        parallel_strategy=ps,
        name=f'conv'
    )
    y = lbann.L2Norm2(x)
    layers = list(lbann.traverse_layer_graph(x))
    metric = lbann.Metric(y, name='obj')
    callbacks = []

    # Compute expected value with NumPy
    vals = []
    for i in range(len(test_dataset)):
        x = test_dataset.samples[i,:].astype(np.float64)
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

    dataset_path = os.path.join(work_dir, 'dataset.pkl')

    # Note: The training data reader should be removed when
    # https://github.com/LLNL/lbann/issues/1098 is resolved.
    message = lbann.reader_pb2.DataReader()
    message.reader.extend([
        construct_python_dataset_reader(
            test_dataset,
            dataset_path,
            'train',
            shuffle=False
        )
    ])
    message.reader.extend([
        construct_python_dataset_reader(
            test_dataset,
            dataset_path,
            'train',
            shuffle=False
        )
    ])
    return message

# ==============================================
# Setup PyTest
# ==============================================

work_dir = os.path.join(os.path.dirname(__file__),
                        'experiments',
                        os.path.basename(__file__).split('.py')[0])
os.makedirs(work_dir, exist_ok=True)

# Create test functions that can interact with PyTest
for _test_func in tools.create_tests(setup_experiment, __file__, work_dir=work_dir,
                                     environment=lbann.contrib.args.get_distconv_environment(
                                         num_io_partitions=tools.gpus_per_node(lbann)
                                     )):
    globals()[_test_func.__name__] = _test_func

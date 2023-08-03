import numpy as np
import lbann
import lbann.modules
from util import preprocess_data

# Data paths, directory where patches are located
data_dir = 'data'
samples = preprocess_data(data_dir)

dims = len(samples[0])


num_classes = 3
num_channels = 14

# Sample access functions
def get_sample(index):
    sample = samples[index]
    return sample

def num_samples():
    return samples.shape[0]

def sample_dims():
    return [dims]

# ==============================================
# Setup and launch experiment
# ==============================================

def construct_model():
    """Model description

    """
    import lbann
    import lbann.modules


    fc = lbann.modules.FullyConnectedModule
    conv = lbann.modules.Convolution2dModule

    conv1 = conv(20, 3, stride=1, padding=1,name='conv1')
    conv2 = conv(20, 3, stride=1, padding=1,name='conv2')
    fc1 = fc(100, name='fc1')
    fc2 = fc(20, name='fc2')
    fc3 = fc(num_classes, name='fc3')
    # Layer graph
    input = lbann.Input(name='inp_tensor', target_mode = 'classification')
    inp_slice = lbann.Slice(input, axis=0, slice_points=[0, dims-1, dims], name='inp_slice')
    xdata = lbann.Identity(inp_slice)
    ylabel = lbann.Identity(inp_slice, name='gt_y')
    #NHWC to NCHW
    x = lbann.Reshape(xdata, dims=[14, 13, 13])
    x = conv2(conv1(x))
    x = lbann.Reshape(x, dims=[3380])
    x = lbann.Dropout(lbann.Relu(fc1(x)),keep_prob=0.5)
    x = lbann.Dropout(fc2(x),keep_prob=0.5)
    pred = lbann.Softmax(fc3(x))
    gt_label  = lbann.OneHot(ylabel, size=num_classes)
    loss = lbann.CrossEntropy([pred,gt_label],name='loss')
    acc = lbann.CategoricalAccuracy([pred, gt_label])


    layers = list(lbann.traverse_layer_graph(input))
    # Setup objective function
    weights = set()
    for l in layers:
      weights.update(l.weights)
    obj = lbann.ObjectiveFunction(loss)


    callbacks = [lbann.CallbackPrint(),
                 lbann.CallbackTimer()]

    # Construct model
    num_epochs = 10
    return lbann.Model(num_epochs,
                       weights=weights,
                       layers=layers,
                       metrics=[lbann.Metric(acc, name='accuracy', unit='%')],
                       objective_function=obj,
                       callbacks=callbacks)

def construct_data_reader():
    """Construct Protobuf message for Python data reader.

    The Python data reader will import this Python file to access the
    sample access functions.

    """
    import os.path
    import lbann
    module_file = os.path.abspath(__file__)
    module_name = os.path.splitext(os.path.basename(module_file))[0]
    module_dir = os.path.dirname(module_file)

    # Base data reader message
    message = lbann.reader_pb2.DataReader()

    # Training set data reader
    data_reader = message.reader.add()
    data_reader.name = 'python'
    data_reader.role = 'train'
    data_reader.shuffle = True
    data_reader.fraction_of_data_to_use = 1.0
    data_reader.python.module = module_name
    data_reader.python.module_dir = module_dir
    data_reader.python.sample_function = 'get_sample'
    data_reader.python.num_samples_function = 'num_samples'
    data_reader.python.sample_dims_function = 'sample_dims'

    return message

if __name__ == '__main__':
    import lbann
    import lbann.contrib.launcher
    mini_batch_size = 64
    trainer = lbann.Trainer(mini_batch_size=mini_batch_size)
    model = construct_model()
    opt = lbann.Adam(learn_rate=0.001,beta1=0.9,beta2=0.99,eps=1e-8)
    data_reader = construct_data_reader()
    status = lbann.contrib.launcher.run(
        trainer, model, data_reader, opt,
        account='hpcdl',
        scheduler='slurm',
        time_limit=720,
        nodes=1,
        procs_per_node=1,
        setup_only=False,
        job_name='candle_p2_ras_classifier')
    print(status)

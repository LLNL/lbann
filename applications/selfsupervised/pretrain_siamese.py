#!/usr/bin/env python3
import functools
import operator
import os.path
import google.protobuf.text_format as txtf
import lbann
import modules
import patch_generator

def setup(num_patches=3,
          mini_batch_size=512,
          num_epochs=75,
          learning_rate=0.005,
          bn_statistics_group_size=2,
          fc_data_layout='model_parallel',
          warmup=True,
          checkpoint_interval=None):

    # Data dimensions
    patch_dims = patch_generator.patch_dims
    num_labels = patch_generator.num_labels(num_patches)

    # Extract tensors from data sample
    input = lbann.Input(data_field='samples')
    slice_points = [0]
    for _ in range(num_patches):
        patch_size = functools.reduce(operator.mul, patch_dims)
        slice_points.append(slice_points[-1] + patch_size)
    slice_points.append(slice_points[-1] + num_labels)
    sample = lbann.Slice(input, slice_points=slice_points)
    patches = [lbann.Reshape(sample, dims=patch_dims)
               for _ in range(num_patches)]
    labels = lbann.Identity(sample)

    # Siamese network
    head_cnn = modules.ResNet(bn_statistics_group_size=bn_statistics_group_size)
    heads = [head_cnn(patch) for patch in patches]
    heads_concat = lbann.Concatenation(heads)

    # Classification network
    class_fc1 = modules.FcBnRelu(4096,
                                 statistics_group_size=bn_statistics_group_size,
                                 name='siamese_class_fc1',
                                 data_layout=fc_data_layout)
    class_fc2 = modules.FcBnRelu(4096,
                                 statistics_group_size=bn_statistics_group_size,
                                 name='siamese_class_fc2',
                                 data_layout=fc_data_layout)
    class_fc3 = lbann.modules.FullyConnectedModule(num_labels,
                                                   activation=lbann.Softmax,
                                                   name='siamese_class_fc3',
                                                   data_layout=fc_data_layout)
    x = class_fc1(heads_concat)
    x = class_fc2(x)
    probs = class_fc3(x)

    # Setup objective function
    cross_entropy = lbann.CrossEntropy([probs, labels])
    l2_reg_weights = set()
    for l in lbann.traverse_layer_graph(input):
        if type(l) == lbann.Convolution or type(l) == lbann.FullyConnected:
            l2_reg_weights.update(l.weights)
    l2_reg = lbann.L2WeightRegularization(weights=l2_reg_weights, scale=0.0002)
    obj = lbann.ObjectiveFunction([cross_entropy, l2_reg])

    # Setup model
    metrics = [lbann.Metric(lbann.CategoricalAccuracy([probs, labels]),
                            name='accuracy', unit='%')]
    callbacks = [lbann.CallbackPrint(), lbann.CallbackTimer()]
    if checkpoint_interval:
        callbacks.append(
            lbann.CallbackCheckpoint(
                checkpoint_dir='ckpt',
                checkpoint_epochs=5
            )
        )

    # Learning rate schedules
    if warmup:
        callbacks.append(
            lbann.CallbackLinearGrowthLearningRate(
                target=learning_rate * mini_batch_size / 128,
                num_epochs=5
            )
        )
    callbacks.append(
        lbann.CallbackDropFixedLearningRate(
            drop_epoch=list(range(0, 100, 15)), amt=0.25)
    )

    # Construct model
    model = lbann.Model(num_epochs,
                        layers=lbann.traverse_layer_graph(input),
                        objective_function=obj,
                        metrics=metrics,
                        callbacks=callbacks)

    # Setup optimizer
    opt = lbann.SGD(learn_rate=learning_rate, momentum=0.9)
    # opt = lbann.Adam(learn_rate=learning_rate, beta1=0.9, beta2=0.999, eps=1e-8)

    # Setup data reader
    data_reader = make_data_reader(num_patches)

    # Return experiment objects
    return model, data_reader, opt

def make_data_reader(num_patches):
    message = lbann.reader_pb2.DataReader()
    data_reader = message.reader.add()
    data_reader.name = 'python'
    data_reader.role = 'train'
    data_reader.shuffle = True
    data_reader.fraction_of_data_to_use = 1.0
    data_reader.python.module = 'patch_generator'
    data_reader.python.module_dir = os.path.dirname(os.path.realpath(__file__))
    data_reader.python.num_samples_function = 'num_samples'
    if num_patches == 2:
        data_reader.python.sample_function = 'get_sample_2patch'
        data_reader.python.sample_dims_function = 'sample_dims_2patch'
    if num_patches == 3:
        data_reader.python.sample_function = 'get_sample_3patch'
        data_reader.python.sample_dims_function = 'sample_dims_3patch'
    if num_patches == 4:
        data_reader.python.sample_function = 'get_sample_4patch'
        data_reader.python.sample_dims_function = 'sample_dims_4patch'
    if num_patches == 5:
        data_reader.python.sample_function = 'get_sample_5patch'
        data_reader.python.sample_dims_function = 'sample_dims_5patch'
    return message

if __name__ == "__main__":
    import argparse
    import lbann.contrib.args
    import lbann.contrib.launcher

    # Command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--job-name', action='store', default='lbann_siamese', type=str,
        help='scheduler job name  (default: lbann_siamese)', metavar='NAME')
    parser.add_argument(
        '--num-patches', action='store', default=3, type=int,
        help='number of patches and Siamese heads (default: 3)', metavar='NUM')
    lbann.contrib.args.add_scheduler_arguments(parser)
    parser.add_argument(
        '--mini-batch-size', action='store', default=512, type=int,
        help='mini-batch size (default: 512)', metavar='NUM')
    parser.add_argument(
        '--num-epochs', action='store', default=75, type=int,
        help='number of epochs (default: 75)', metavar='NUM')
    parser.add_argument(
        '--learning-rate', action='store', default=0.005, type=float,
        help='learning rate (default: 0.005)', metavar='LR')
    parser.add_argument(
        '--bn-statistics-group-size', action='store', default=2, type=int,
        help=('group size for batch norm statistics (default: 2)'))
    parser.add_argument(
        '--fc-data-layout', action='store', default='model_parallel', type=str,
        help=('data layout for fully-connected layers '
              '(default: "model_parallel")'))
    parser.add_argument(
        '--warmup', action='store', default=True, type=bool,
        help='use learning rate warmup (default: True)')
    args = parser.parse_args()

    # Setup experiment
    trainer = lbann.Trainer(mini_batch_size=args.mini_batch_size)
    model, data_reader, opt = setup(
        num_patches=args.num_patches,
        mini_batch_size=args.mini_batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        bn_statistics_group_size=args.bn_statistics_group_size,
        fc_data_layout=args.fc_data_layout,
        warmup=args.warmup,
    )

    # Run experiment
    kwargs = lbann.contrib.args.get_scheduler_kwargs(args)
    lbann.contrib.launcher.run(
        trainer, model, data_reader, opt,
        job_name = args.job_name,
        **kwargs,
    )

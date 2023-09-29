#!/usr/bin/env python3
import os.path
import google.protobuf.text_format
import lbann
import modules

def setup(data_reader_file,
          name='classifier',
          num_labels=200,
          mini_batch_size=128,
          num_epochs=1000,
          learning_rate=0.1,
          bn_statistics_group_size=2,
          fc_data_layout='model_parallel',
          warmup_epochs=50,
          learning_rate_drop_interval=50,
          learning_rate_drop_factor=0.25,
          checkpoint_interval=None):

    # Setup input data
    images = lbann.Input(data_field='samples')
    labels = lbann.Input(data_field='labels')

    # Classification network
    head_cnn = modules.ResNet(bn_statistics_group_size=bn_statistics_group_size)
    class_fc = lbann.modules.FullyConnectedModule(num_labels,
                                                  activation=lbann.Softmax,
                                                  name=f'{name}_fc',
                                                  data_layout=fc_data_layout)
    x = head_cnn(images)
    probs = class_fc(x)

    # Setup objective function
    cross_entropy = lbann.CrossEntropy([probs, labels])
    l2_reg_weights = set()
    for l in lbann.traverse_layer_graph([images, labels]):
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
    if warmup_epochs:
        callbacks.append(
            lbann.CallbackLinearGrowthLearningRate(
                target=learning_rate * mini_batch_size / 128,
                num_epochs=warmup_epochs
            )
        )
    if learning_rate_drop_factor:
        callbacks.append(
            lbann.CallbackDropFixedLearningRate(
                drop_epoch=list(range(0, num_epochs, learning_rate_drop_interval)),
                amt=learning_rate_drop_factor)
        )

    # Construct model
    model = lbann.Model(num_epochs,
                        layers=lbann.traverse_layer_graph(input),
                        objective_function=obj,
                        metrics=metrics,
                        callbacks=callbacks)

    # Setup optimizer
    # opt = lbann.Adam(learn_rate=learning_rate, beta1=0.9, beta2=0.999, eps=1e-8)
    opt = lbann.SGD(learn_rate=learning_rate, momentum=0.9)

    # Load data reader from prototext
    data_reader_proto = lbann.lbann_pb2.LbannPB()
    with open(data_reader_file, 'r') as f:
        google.protobuf.text_format.Merge(f.read(), data_reader_proto)
    data_reader_proto = data_reader_proto.data_reader
    for reader_proto in data_reader_proto.reader:
        reader_proto.python.module_dir = os.path.dirname(os.path.realpath(__file__))

    # Return experiment objects
    return model, data_reader_proto, opt

if __name__ == "__main__":
    import argparse
    import lbann.contrib.args
    import lbann.contrib.launcher

    # Command-line arguments
    parser = argparse.ArgumentParser()
    lbann.contrib.args.add_scheduler_arguments(parser, 'lbann_siamese_finetune')
    parser.add_argument(
        '--mini-batch-size', action='store', default=128, type=int,
        help='mini-batch size (default: 128)', metavar='NUM')
    parser.add_argument(
        '--num-epochs', action='store', default=1000, type=int,
        help='number of epochs (default: 1000)', metavar='NUM')
    parser.add_argument(
        '--learning-rate', action='store', default=0.1, type=float,
        help='learning rate (default: 0.1)', metavar='LR')
    parser.add_argument(
        '--bn-statistics-group-size', action='store', default=2, type=int,
        help=('group size for batch norm statistics (default: 2)'))
    parser.add_argument(
        '--fc-data-layout', action='store', default='model_parallel', type=str,
        help=('data layout for fully-connected layers '
              '(default: "model_parallel")'))
    args = parser.parse_args()

    # Setup experiment
    trainer = lbann.Trainer(mini_batch_size=args.mini_batch_size)
    current_dir = os.path.dirname(os.path.realpath(__file__))
    data_reader_file = os.path.join(current_dir, 'data_reader_cub.prototext')
    model, data_reader, opt = setup(
        data_reader_file=data_reader_file,
        mini_batch_size=args.mini_batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        bn_statistics_group_size=args.bn_statistics_group_size,
        fc_data_layout=args.fc_data_layout,
    )

    # Run experiment
    kwargs = lbann.contrib.args.get_scheduler_kwargs(args)
    lbann.contrib.launcher.run(
        trainer, model, data_reader, opt,
        job_name=args.job_name,
        **kwargs,
    )

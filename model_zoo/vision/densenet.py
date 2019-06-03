#!/usr/bin/env python3
import argparse
import os.path
import subprocess
import google.protobuf.text_format as txtf
import lbann
import lbann.contrib.args
import lbann.contrib.lc.launcher

# TODO: Add trainer argument after PR #916 merges

LOG = True


def log(string):
    if LOG:
        print(string)

# Commands to run ##############################################################

# Allocate notes on Pascal from ssh:
# salloc --nodes=16 --partition=pbatch --time=180

# From lbann/model_zoo/vision:
# ./densenet.py
# --disable-run (if experiment shouldn't be run)
# --mini-batch-size 128 (if mini-batch-size should be something other than 256)
# --nodes 16 (if more than one node is to be used; 16 is optimal)
# --procs-per-node 2

# To run the full 90 epochs from ssh:
# ./densenet.py --nodes 16 --procs-per-node 2 > /usr/workspace/wsb/<username>/lbann/model_zoo/vision/output.txt
# mini-batch-size default => 256, num-epochs => 90

# To run 10 epoch test from ssh:
# ./densenet.py --nodes 16 --procs-per-node 2 --mini-batch-size 256 --num-epochs 10 > /usr/workspace/wsb/<username>/lbann/model_zoo/vision/output.txt

# To avoid needing to stay logged into ssh, create a script
# densenet_batch_job.cmd such as:
# #!/bin/bash
# #SBATCH --nodes 16
# #SBATCH --partition pbatch
# #SBATCH --time 240
# ./densenet.py --nodes 16 --procs-per-node 2 --mini-batch-size 256 --num-epochs 10 > /usr/workspace/wsb/<username>/lbann/model_zoo/vision/output.txt

# and from lbann/model_zoo/vision run:
# sbatch densenet_batch_job.cmd

# To generate visualization, from lbann run:
# scripts/viz.py model_zoo/models/densenet/generated_densenet.prototext

# Copy the output file, experiment directory, and visualization
# from LC to your computer by running the following commands from your computer:
# scp <username>@pascal.llnl.gov:/usr/workspace/wsb/<username>/lbann/model_zoo/vision/output.txt .
# scp -r <username>@pascal.llnl.gov:/usr/workspace/wsb/<username>/lbann/experiments/<date_time>_lbann_densenet/ .
# scp <username>@pascal.llnl.gov:/usr/workspace/wsb/<username>/lbann/graph.pdf .


# DenseNet #####################################################################
# See src/proto/lbann.proto for possible functions to call.
# See PyTorch DenseNet:
# https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py
# See "Densely Connected Convolutional Networks" by Huang et. al p.4
def densenet(version,
             cumulative_layer_num,
             images_node
             ):
    if version == 121:
        growth_rate = 32  # k in the paper
        layers_per_block = (6, 12, 24, 16)
        num_initial_features = 64
    elif version == 161:
        growth_rate = 48  # k in the paper
        layers_per_block = (96, 48, 36, 24)
        num_initial_features = 96
    else:
        raise Exception('Invalid version={v}.'.format(v=version))
    batch_norm_size = 4

    parent_node, cumulative_layer_num = initial_layer(
        cumulative_layer_num, images_node,
        num_initial_features)
    num_features = num_initial_features
    # Start counting dense blocks at 1.
    for current_block_num, num_layers in enumerate(layers_per_block, 1):
        parent_nodes, cumulative_layer_num = dense_block(
            cumulative_layer_num,
            parent_node,
            batch_norm_size=batch_norm_size,
            current_block_num=current_block_num,
            growth_rate=growth_rate,
            num_layers=num_layers,
            num_initial_channels=num_initial_features
        )
        # num_features += num_layers * growth_rate
        for node in parent_nodes[1:]:
            num_features += node.num_output_channels
        parent_node = lbann.Concatenation(parent_nodes)
        cumulative_layer_num += 1
        log('densenet Concatenation. cumulative_layer_num={n}'.format(
            b=current_block_num, n=cumulative_layer_num))
        if current_block_num != len(layers_per_block):
            parent_node, cumulative_layer_num = transition_layer(
                current_block_num,
                cumulative_layer_num,
                parent_node,
                # In Python 3, this is integer division.
                num_output_channels=num_features//2,
            )
            num_features //= 2

    batch_normalization_node = standard_batchnorm(parent_node)
    cumulative_layer_num += 1
    log('densenet BatchNormalization. cumulative_layer_num={n}'.format(
        b=current_block_num, n=cumulative_layer_num))

    probs = classification_layer(
        cumulative_layer_num,
        batch_normalization_node
    )
    return probs


def initial_layer(cumulative_layer_num,
                  images_node,
                  num_initial_channels
                  ):
    # 7x7 conv, stride 2
    convolution_node = lbann.Convolution(
        images_node,
        conv_dims_i=7,
        conv_pads_i=3,
        conv_strides_i=2,
        has_bias=False,
        num_dims=2,
        num_output_channels=num_initial_channels
    )
    cumulative_layer_num += 1
    log('initial_layer Convolution. cumulative_layer_num={n}'.format(
        n=cumulative_layer_num))

    batch_normalization_node = standard_batchnorm(convolution_node)
    cumulative_layer_num += 1
    log('initial_layer BatchNormalization. cumulative_layer_num={n}'.format(
        n=cumulative_layer_num))

    relu_node = lbann.Relu(batch_normalization_node)
    cumulative_layer_num += 1
    log('initial_layer Relu. cumulative_layer_num={n}'.format(
        n=cumulative_layer_num))

    # 3x3 max pool, stride 2
    pooling_node = lbann.Pooling(
        relu_node,
        num_dims=2,
        pool_dims_i=3,
        pool_mode='max',
        pool_pads_i=1,
        pool_strides_i=2
        )
    cumulative_layer_num += 1
    log('initial_layer Pooling. cumulative_layer_num={n}'.format(
        n=cumulative_layer_num))

    return pooling_node, cumulative_layer_num


def standard_batchnorm(parent_node):
    return lbann.BatchNormalization(
        parent_node,
        bias_init=0.0,
        decay=0.9,
        epsilon=1e-5,
        scale_init=1.0
    )


def dense_block(cumulative_layer_num,
                parent_node,
                batch_norm_size,
                current_block_num,
                growth_rate,
                num_layers,
                num_initial_channels
                ):
    parent_nodes = [parent_node]
    # Start counting dense layers at 1.
    for current_layer_num in range(1, num_layers + 1):
        # channels from before block + (each dense layer has k=growth_rate channels)
        num_input_channels = num_initial_channels + (current_layer_num - 1) * growth_rate
        print('num_input_channels={c}'.format(c=num_input_channels))
        parent_node, cumulative_layer_num = dense_layer(
            current_block_num,
            current_layer_num,
            cumulative_layer_num,
            parent_nodes,
            batch_norm_size=batch_norm_size,
            growth_rate=growth_rate
        )
        parent_nodes.append(parent_node)
    return parent_nodes, cumulative_layer_num


def dense_layer(current_block_num,
                current_layer_num,
                cumulative_layer_num,
                parent_nodes,
                batch_norm_size,
                growth_rate
                ):
    concatenation_node = lbann.Concatenation(parent_nodes)
    cumulative_layer_num += 1
    log('dense_block={b} dense_layer={l} Concatenation. cumulative_layer_num={n}'.format(
        b=current_block_num, l=current_layer_num, n=cumulative_layer_num))
    conv_block_1_node, cumulative_layer_num = conv_block(
        current_block_num,
        current_layer_num,
        cumulative_layer_num,
        concatenation_node,
        conv_dims_i=1,
        conv_pads_i=0,
        num_output_channels=batch_norm_size * growth_rate
    )
    conv_block_2_node, cumulative_layer_num = conv_block(
        current_block_num,
        current_layer_num,
        cumulative_layer_num,
        conv_block_1_node,
        conv_dims_i=3,
        conv_pads_i=1,
        num_output_channels=growth_rate
    )
    return conv_block_2_node, cumulative_layer_num


def conv_block(current_block_num,
               current_layer_num,
               cumulative_layer_num,
               parent_node,
               conv_dims_i,
               conv_pads_i,
               num_output_channels
               ):
    batch_normalization_node = standard_batchnorm(parent_node)
    cumulative_layer_num += 1
    log('dense_block={b} dense_layer={l} BatchNormalization. cumulative_layer_num={n}'.format(
        b=current_block_num, l=current_layer_num, n=cumulative_layer_num))

    relu_node = lbann.Relu(batch_normalization_node)
    cumulative_layer_num += 1
    log(
        'dense_block={b} dense_layer={l} Relu. cumulative_layer_num={n}'.format(
            b=current_block_num, l=current_layer_num, n=cumulative_layer_num))

    convolution_node = lbann.Convolution(
        relu_node,
        conv_dims_i=conv_dims_i,
        conv_pads_i=conv_pads_i,
        conv_strides_i=1,
        has_bias=False,
        num_dims=2,
        num_output_channels=num_output_channels
    )
    cumulative_layer_num += 1
    log('dense_block={b} dense_layer={l} Convolution. cumulative_layer_num={n}'.format(
        b=current_block_num, l=current_layer_num, n=cumulative_layer_num))

    return convolution_node, cumulative_layer_num


def transition_layer(current_block_num,
                     cumulative_layer_num,
                     parent_node,
                     num_output_channels
                     ):
    batch_normalization_node = standard_batchnorm(parent_node)
    cumulative_layer_num += 1
    log('dense_block={b} > transition_layer BatchNormalization. cumulative_layer_num={n}'.format(
        b=current_block_num,  n=cumulative_layer_num))

    relu_node = lbann.Relu(batch_normalization_node)
    cumulative_layer_num += 1
    log('dense_block={b} > transition_layer Relu. cumulative_layer_num={n}'.format(
        b=current_block_num, n=cumulative_layer_num))

    convolution_node = lbann.Convolution(
        relu_node,
        conv_dims_i=1,
        conv_pads_i=0,
        conv_strides_i=1,
        has_bias=False,
        num_dims=2,
        num_output_channels=num_output_channels
    )
    cumulative_layer_num += 1
    log('dense_block={b} > transition_layer Convolution. cumulative_layer_num={n}'.format(
        b=current_block_num, n=cumulative_layer_num))

    # 2x2 average pool, stride 2
    pooling_node = lbann.Pooling(
        convolution_node,
        num_dims=2,
        pool_dims_i=2,
        pool_mode='average',
        pool_pads_i=0,
        pool_strides_i=2
    )
    cumulative_layer_num += 1
    log('dense_block={b} > transition_layer Pooling. cumulative_layer_num={n}'.format(
        b=current_block_num, n=cumulative_layer_num))

    return pooling_node, cumulative_layer_num


def classification_layer(cumulative_layer_num,
                         parent_node):
    # 7x7 global average pool
    pooling_node = lbann.Pooling(
        parent_node,
        num_dims=2,
        pool_dims_i=7,
        pool_mode='average',
        pool_pads_i=1,
        pool_strides_i=1
    )
    cumulative_layer_num += 1
    log('classification_layer Pooling. cumulative_layer_num={n}'.format(
        n=cumulative_layer_num))

    fully_connected_node = lbann.FullyConnected(
        pooling_node,
        num_neurons=1000,
        has_bias=False
    )
    cumulative_layer_num += 1
    log('classification_layer FullyConnected. cumulative_layer_num={n}'.format(
        n=cumulative_layer_num))

    probabilities = lbann.Softmax(fully_connected_node)
    return probabilities


# Helpful Functions ############################################################
def get_args():
    desc = ('Construct and run DenseNet on ImageNet data. '
            'Running the experiment is only supported on LC systems.')
    parser = argparse.ArgumentParser(description=desc)
    lbann.contrib.args.add_scheduler_arguments(parser)
    parser.add_argument(
        '--mini-batch-size', action='store', default=256, type=int,
        help='mini-batch size (default: 256)', metavar='NUM')
    parser.add_argument(
        '--num-epochs', action='store', default=90, type=int,
        help='number of epochs (default: 90)', metavar='NUM')
    parser.add_argument(
        '--num-labels', action='store', default=1000, type=int,
        help='number of data classes (default: 1000)', metavar='NUM')
    lbann.contrib.args.add_optimizer_arguments(
        parser,
        default_optimizer='sgd',
        default_learning_rate=0.1
    )
    lbann_dir = subprocess.check_output(
        'git rev-parse --show-toplevel'.split()).strip()
    # https://stackoverflow.com/questions/606191/convert-bytes-to-a-string
    lbann_dir = lbann_dir.decode("utf-8")
    data_reader_prototext = os.path.join(lbann_dir,
                                         'model_zoo',
                                         'data_readers',
                                         'data_reader_imagenet.prototext')
    parser.add_argument(
        '--data-reader', action='store',
        default=data_reader_prototext, type=str,
        help='data reader prototext file (default: ' + data_reader_prototext + ')',
        metavar='FILE')
    parser.add_argument(
        '--imagenet-classes', action='store', type=int,
        help='number of ImageNet-1K classes (availability of subsampled datasets may vary by system)',
        metavar='NUM')
    generated_prototext = os.path.join(lbann_dir,
                                       'model_zoo',
                                       'models',
                                       'densenet',
                                       'generated_densenet.prototext')
    parser.add_argument(
        '--prototext', action='store',
        default=generated_prototext, type=str,
        help='exported prototext file', metavar='FILE')
    parser.add_argument(
        '--disable-run', action='store_true',
        help='do not run experiment (e.g. if only the prototext is desired)')
    args = parser.parse_args()
    return args


def construct_layer_graph(
        version,
        cumulative_layer_num,
        input_node):
    # Input data
    images_node = lbann.Identity(input_node)
    cumulative_layer_num += 1
    log('Identity. cumulative_layer_num={n}'.format(n=cumulative_layer_num))

    # Use input_node, not images_node.
    image_labels_node = lbann.Identity(input_node)
    cumulative_layer_num += 1
    log('Identity. cumulative_layer_num={n}'.format(n=cumulative_layer_num))

    # Use images_node, not image_labels_node.
    probabilities = densenet(version, cumulative_layer_num, images_node)

    return probabilities, image_labels_node


def set_up_experiment(args,
                      input_,
                      probs,
                      labels):
    # Set up objective function
    cross_entropy = lbann.CrossEntropy([probs, labels])
    layers = list(lbann.traverse_layer_graph(input_))
    weights = set()
    for l in layers:
        weights.update(l.weights)
    # scale = weight decay
    l2_reg = lbann.L2WeightRegularization(weights=weights, scale=1e-4)
    objective_function = lbann.ObjectiveFunction([cross_entropy, l2_reg])

    # Set up model
    top1 = lbann.CategoricalAccuracy([probs, labels])
    top5 = lbann.TopKCategoricalAccuracy([probs, labels], k=5)
    metrics = [lbann.Metric(top1, name='top-1 accuracy', unit='%'),
               lbann.Metric(top5, name='top-5 accuracy', unit='%')]
    callbacks = [lbann.CallbackPrint(),
                 lbann.CallbackTimer(),
                 lbann.CallbackDropFixedLearningRate(
                     drop_epoch=[30, 60], amt=0.1)]
    model = lbann.Model(args.mini_batch_size,
                        args.num_epochs,
                        layers=layers,
                        weights=weights,
                        objective_function=objective_function,
                        metrics=metrics,
                        callbacks=callbacks)

    # Load data reader from prototext
    data_reader_proto = lbann.lbann_pb2.LbannPB()
    with open(args.data_reader, 'r') as f:
        txtf.Merge(f.read(), data_reader_proto)
    data_reader_proto = data_reader_proto.data_reader

    # Set up optimizer
    if args.optimizer == 'sgd':
        print('Creating sgd optimizer')
        optimizer = lbann.optimizer.SGD(
            learn_rate=args.optimizer_learning_rate,
            momentum=0.9,
            nesterov=True
        )
    else:
        optimizer = lbann.contrib.args.create_optimizer(args)

    # Save prototext to args.prototext
    if args.prototext:
        lbann.proto.save_prototext(args.prototext,
                                   model=model,
                                   optimizer=optimizer,
                                   data_reader=data_reader_proto)

    return model, data_reader_proto, optimizer


def run_experiment(args,
                   model,
                   data_reader_proto,
                   optimizer):
    # Run experiment
    if not args.disable_run:
        from lbann.contrib.lc.paths import imagenet_dir, imagenet_labels
        import lbann.contrib.lc.launcher
        kwargs = {}
        if args.nodes:
            kwargs['nodes'] = args.nodes
        if args.procs_per_node:
            kwargs['procs_per_node'] = args.procs_per_node
        if args.partition:
            kwargs['partition'] = args.partition
        if args.account:
            kwargs['account'] = args.account
        if args.time_limit:
            kwargs['time_limit'] = args.time_limit
        if args.imagenet_classes:
            classes = args.imagenet_classes
            kwargs['lbann_args'] = (
                '--data_filedir_train={} --data_filename_train={} '
                '--data_filedir_test={} --data_filename_test={}'
                    .format(imagenet_dir(data_set='train', num_classes=classes),
                            imagenet_labels(data_set='train',
                                            num_classes=classes),
                            imagenet_dir(data_set='val', num_classes=classes),
                            imagenet_labels(data_set='val',
                                            num_classes=classes)))
        lbann.contrib.lc.launcher.run(model,
                                      data_reader_proto,
                                      optimizer,
                                      job_name='lbann_densenet',
                                      **kwargs)


# Main function ################################################################
def main():
    # ----------------------------------
    # Command-line arguments
    # ----------------------------------
    args = get_args()

    # ----------------------------------
    # Construct layer graph
    # ----------------------------------
    input_node = lbann.Input()
    # Start counting cumulative layers at 1.
    cumulative_layer_num = 1
    log('Input. cumulative_layer_num={n}'.format(n=cumulative_layer_num))
    (probs, labels) = construct_layer_graph(
        121, cumulative_layer_num, input_node)

    # ----------------------------------
    # Setup experiment
    # ----------------------------------

    (model, data_reader_proto, optimizer) = set_up_experiment(
        args, input_node, probs, labels)

    # ----------------------------------
    # Run experiment
    # ----------------------------------
    # Note: Use `lbann.run` instead for non-LC systems.

    run_experiment(args, model, data_reader_proto, optimizer)


if __name__ == '__main__':
    main()

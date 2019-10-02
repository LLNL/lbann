import sys
sys.path.insert(0, '../common_python')
import tools

import os
import pytest


def skeleton_mnist_ridge_regression(cluster, executables, dir_name,
                                    compiler_name, weekly, data_reader_percent):
    tools.process_executable(
       'skeleton_mnist_ridge_regression', compiler_name, executables)

    if compiler_name == 'exe':
        exe = executables[compiler_name]
        bin_dir = os.path.dirname(exe)
        install_dir = os.path.dirname(bin_dir)
        build_path = '{i}/lib/python3.7/site-packages'.format(i=install_dir)
    else:
        if compiler_name == 'clang6':
            path = 'clang.Release'
        elif compiler_name == 'clang6_debug':
            path = 'clang.Debug'
        elif compiler_name == 'gcc7':
            path = 'gnu.Release'
        elif compiler_name == 'clang6_debug':
            path = 'gnu.Debug'
        elif compiler_name == 'intel19':
            path = 'intel.Release'
        elif compiler_name == 'intel19_debug':
            path = 'intel.Debug'
        path = '{p}.{c}.llnl.gov'.format(p=path, c=cluster)
        build_path = '{d}/build/{p}/install/lib/python3.7/site-packages'.format(
            d=dir_name, p=path)
    print('build_path={b}'.format(b=build_path))
    sys.path.append(build_path)

    # Model
    # Converted from lbann/model_zoo/tests/model_mnist_ridge_regression.prototext.
    # Equivalent to prototext's "Layers" section.
    import lbann
    input_node = lbann.Input()
    images_node = lbann.Identity(input_node)
    image_labels_node = lbann.Identity(input_node)
    fc_node = lbann.FullyConnected(images_node, num_neurons=10, has_bias=True)
    mse = lbann.MeanSquaredError([fc_node, image_labels_node])
    # Equivalent to prototext's "Objective function" section.
    layers = list(lbann.traverse_layer_graph(input_node))
    weights = set()
    for l in layers:
        weights.update(l.weights)
    # scale == weight decay
    l2_reg = lbann.L2WeightRegularization(weights=weights, scale=0.01)
    objective_function = lbann.ObjectiveFunction([mse, l2_reg])
    # Equivalent to prototext's "Metrics" section.
    metrics = [lbann.Metric(mse, name='mean squared error')]
    # Equivalent to prototext's "Callbacks" section.
    callbacks = [lbann.CallbackPrint(),
                 lbann.CallbackTimer(),
                 lbann.CallbackCheckGradients(
                     verbose=False, error_on_failure=True)]
    # Equivalent to prototext's model-level parameters.
    model = lbann.Model(mini_batch_size=131,
                        epochs=4,
                        layers=layers,
                        objective_function=objective_function,
                        metrics=metrics,
                        callbacks=callbacks)

    # Data Reader
    # TODO: Do we also want to programatically construct the data reader, not just the model?
    data_reader_prototext_file = os.path.join(dir_name,
                                              'model_zoo',
                                              'data_readers',
                                              'data_reader_mnist.prototext')
    data_reader_proto = lbann.lbann_pb2.LbannPB()
    with open(data_reader_prototext_file, 'r') as f:
        import google.protobuf.text_format as txtf
        txtf.Merge(f.read(), data_reader_proto)
    data_reader_proto = data_reader_proto.data_reader

    # Optimizer
    # Learning rate from model_zoo/optimizers/opt_adam.prototext
    optimizer = lbann.optimizer.Adam(learn_rate=0.001, beta1=0.9, beta2=0.99, eps=1e-8)

    # kwargs
    kwargs = {
        'account': 'guests',
        'nodes': 1,
        'partition': 'pbatch',
        'procs_per_node': 1
    }

    if data_reader_percent is None:
        if weekly:
            data_reader_percent = 1.00
        else:
            # Nightly
            data_reader_percent = 0.10
    lbann_args = '--data_reader_percent={drp}'.format(drp=data_reader_percent)
    if cluster == 'lassen':
        lbann_args += ' --data_filedir_train=/p/gpfs1/brainusr/datasets/MNIST --data_filedir_test=/p/gpfs1/brainusr/datasets/MNIST'
    kwargs['lbann_args'] = lbann_args

    # Run
    experiment_dir = '{d}/bamboo/unit_tests/experiments/mnist_ridge_regression_{c}'.format(
        d=dir_name, c=compiler_name)
    # Setup trainer
    trainer = lbann.Trainer()
    import lbann.contrib.lc.launcher
    return_code = lbann.contrib.lc.launcher.run(
        trainer=trainer,
        experiment_dir=experiment_dir,
        model=model,
        data_reader=data_reader_proto,
        optimizer=optimizer,
        job_name='lbann_ridge_regression',
        **kwargs)

    error_file_name = '{e}/err.log'.format(
        e=experiment_dir, c=compiler_name)
    tools.assert_success(return_code, error_file_name)


def test_unit_mnist_ridge_regression_clang6(cluster, exes, dirname,
                                            weekly, data_reader_percent):
    skeleton_mnist_ridge_regression(cluster, exes, dirname, 'clang6',
                                    weekly, data_reader_percent)


def test_unit_mnist_ridge_regression_gcc7(cluster, exes, dirname,
                                          weekly, data_reader_percent):
    skeleton_mnist_ridge_regression(cluster, exes, dirname, 'gcc7',
                                    weekly, data_reader_percent)


def test_unit_mnist_ridge_regression_intel19(cluster, exes, dirname,
                                             weekly, data_reader_percent):
    skeleton_mnist_ridge_regression(cluster, exes, dirname, 'intel19',
                                    weekly, data_reader_percent)


# Run with python3 -m pytest -s test_unit_ridge_regression.py -k 'test_unit_mnist_ridge_regression_exe' --exe=<executable>
def test_unit_mnist_ridge_regression_exe(cluster, dirname, exe,
                                         weekly, data_reader_percent):
    if exe is None:
        e = 'test_unit_mnist_ridge_regression_exe: Non-local testing'
        print('Skip - ' + e)
        pytest.skip(e)
    exes = {'exe': exe}
    skeleton_mnist_ridge_regression(cluster, exes, dirname, 'exe',
                                    weekly, data_reader_percent)

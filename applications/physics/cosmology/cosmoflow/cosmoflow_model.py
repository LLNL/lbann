import lbann
import cosmoflow_network_architectures
import math

def construct_cosmoflow_model(parallel_strategy,
                              local_batchnorm,
                              input_width,
                              num_secrets,
                              use_batchnorm,
                              num_epochs,
                              depth_splits_pooling_id,
                              gather_dropout_id):

    # Construct layer graph
    universes = lbann.Input(data_field='samples')
    secrets = lbann.Input(data_field='responses')
    statistics_group_size = 1 if local_batchnorm else -1
    preds = cosmoflow_network_architectures.CosmoFlow(
        input_width=input_width,
        output_size=num_secrets,
        use_bn=use_batchnorm,
        bn_statistics_group_size=statistics_group_size)(universes)
    mse = lbann.MeanSquaredError([preds, secrets])
    obj = lbann.ObjectiveFunction([mse])
    layers = list(lbann.traverse_layer_graph([universes, secrets]))

    # Set parallel_strategy
    if parallel_strategy is not None:
        pooling_id = 0
        dropout_id = 0
        depth_groups = parallel_strategy['depth_groups']
        for i, layer in enumerate(layers):
            if layer == secrets:
                continue

            layer_name = layer.__class__.__name__
            if layer_name == 'Pooling':
                pooling_id += 1

                if depth_splits_pooling_id is None:
                    assert 2**math.log2(depth_groups) == depth_groups
                    depth_splits_pooling_id = 5-(math.log2(depth_groups)-2)

                if pooling_id == depth_splits_pooling_id:
                    parallel_strategy = dict(parallel_strategy.items())
                    parallel_strategy['depth_splits'] = 1

            elif layer_name == 'Dropout':
                dropout_id += 1
                if dropout_id == gather_dropout_id:
                    break

            layer.parallel_strategy = parallel_strategy

    # Set up model
    metrics = [lbann.Metric(mse, name='MSE', unit='')]
    callbacks = [
        lbann.CallbackPrint(),
        lbann.CallbackTimer(),
        lbann.CallbackGPUMemoryUsage(),
        lbann.CallbackPrintModelDescription(),
        lbann.CallbackDumpOutputs(
            directory='dump_acts/',
            layers=' '.join([preds.name, secrets.name]),
            execution_modes='test'
        ),
        lbann.CallbackProfiler(skip_init=True)]
    base_lr = 1e-3
    for i in range(5):
        fac = 1e-2 + (1 - 1e-2) * i / 4
        callbacks.append(lbann.CallbackSetLearningRate(step=i, val=fac * base_lr))
    callbacks += [
        lbann.CallbackSetLearningRate(step=32, val=0.25 * base_lr),
        lbann.CallbackSetLearningRate(step=64, val=0.125 * base_lr),
    ]
    # # TODO: Use polynomial learning rate decay (https://github.com/LLNL/lbann/issues/1581)
    # callbacks.append(lbann.CallbackPolyLearningRate(
    #     power=1.0,
    #     num_epochs=100,
    #     end_lr=1e-7))
    return lbann.Model(
        epochs=num_epochs,
        layers=layers,
        objective_function=obj,
        metrics=metrics,
        callbacks=callbacks
    )

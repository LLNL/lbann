import lbann
import cosmoflow_network_architectures
import math

def construct_cosmoflow_model(parallel_strategy,
                              local_batchnorm,
                              input_width,
                              num_secrets,
                              use_batchnorm,
                              num_epochs,
                              learning_rate,
                              min_distconv_width,
                              mlperf,
                              transform_input):

    # Construct layer graph
    universes = lbann.Input(data_field='samples')
    secrets = lbann.Input(data_field='responses')
    statistics_group_size = 1 if local_batchnorm else -1
    preds = cosmoflow_network_architectures.CosmoFlow(
        input_width=input_width,
        output_size=num_secrets,
        use_bn=use_batchnorm,
        bn_statistics_group_size=statistics_group_size,
        mlperf=mlperf,
        transform_input=transform_input)(universes)
    mse = lbann.MeanSquaredError([preds, secrets])
    mae = lbann.MeanAbsoluteError([preds, secrets])
    obj = lbann.ObjectiveFunction([mse])
    layers = list(lbann.traverse_layer_graph([universes, secrets]))

    # Set parallel_strategy
    if parallel_strategy is not None:
        depth_groups = int(parallel_strategy['depth_groups'])
        if min_distconv_width is None:
            min_distconv_width = depth_groups
        else:
            min_distconv_width = max(depth_groups, min_distconv_width)
        last_distconv_layer = int(math.log2(input_width)
                                  - math.log2(min_distconv_width) + 1)
        for layer in layers:
            if layer == secrets:
                continue

            if f'pool{last_distconv_layer}' in layer.name or 'fc' in layer.name:
                break

            layer.parallel_strategy = parallel_strategy

    # Set up model
    metrics = [lbann.Metric(mse, name='MSE', unit=''),
               lbann.Metric(mae, name='MAE', unit='')]
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
        lbann.CallbackLinearGrowthLearningRate(target=learning_rate, num_epochs=5),
        lbann.CallbackSetLearningRate(step=32, val=0.25 * learning_rate),
        lbann.CallbackSetLearningRate(step=64, val=0.125 * learning_rate),
        lbann.CallbackProgressBar()
    ]
    return lbann.Model(
        epochs=num_epochs,
        layers=layers,
        objective_function=obj,
        metrics=metrics,
        callbacks=callbacks
    )

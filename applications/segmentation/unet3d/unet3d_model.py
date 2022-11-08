import lbann
import unet3d_network_architectures

def construct_unet3d_model(parallel_strategy,
                           num_epochs):

    # Construct layer graph
    volume = lbann.Input(data_field='samples')
    segmentation = lbann.Input(data_field='label_reconstruction')
    output = unet3d_network_architectures.UNet3D()(volume)
    ce = lbann.CrossEntropy(
        [output, segmentation],
        use_labels=True)
    obj = lbann.ObjectiveFunction([ce])
    layers = list(lbann.traverse_layer_graph([volume, segmentation]))
    for l in layers:
        l.parallel_strategy = parallel_strategy

    # Setup model
    metrics = [lbann.Metric(ce, name='CE', unit='')]
    callbacks = [
        lbann.CallbackPrint(),
        lbann.CallbackTimer(),
        lbann.CallbackPrintModelDescription(),
        lbann.CallbackGPUMemoryUsage(),
        lbann.CallbackProfiler(skip_init=True),
    ]
    # # TODO: Use polynomial learning rate decay (https://github.com/LLNL/lbann/issues/1581)
    # callbacks.append(
    #     lbann.CallbackPolyLearningRate(
    #         power=1.0,
    #         num_epochs=100,
    #         end_lr=1e-5))
    return lbann.Model(
        epochs=num_epochs,
        layers=layers,
        objective_function=obj,
        callbacks=callbacks,
    )

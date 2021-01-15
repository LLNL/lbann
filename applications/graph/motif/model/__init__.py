import lbann
import model.autoencoder

def make_model(
        data_dim,
        latent_dim,
        num_epochs,
):

    # Layer graph
    data = lbann.Identity(lbann.Input())
    autoencoder = model.autoencoder.FullyConnectedAutoencoder(
        data_dim,
        latent_dim,
    )
    reconstructed = autoencoder(data)
    loss = lbann.MeanSquaredError(data, reconstructed)

    # Metrics
    metrics = [
        lbann.Metric(loss, name='loss'),
    ]

    # Callbacks
    callbacks = [
        lbann.CallbackPrint(),
        lbann.CallbackTimer(),
        lbann.CallbackDumpWeights(directory='weights',
                                  epoch_interval=num_epochs),
    ]

    # Contruct model
    return lbann.Model(
        num_epochs,
        layers=lbann.traverse_layer_graph(loss),
        objective_function=loss,
        metrics=metrics,
        callbacks=callbacks,
    )

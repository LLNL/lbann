import lbann
import numpy as np

import model.gan
from util import str_list

def make_model(
        motif_size,
        walk_length,
        num_vertices,
        embed_dim,
        learn_rate,
        num_epochs,
        embeddings_dir,
        online_walker,
        generator_type='greedy',
        embeddings_device='CPU',
        initial_embeddings_file=None,
):

    # Load initial embeddings if provided
    if initial_embeddings_file:
        initial_embeddings = np.loadtxt(initial_embeddings_file, skiprows=1)
    else:
        initial_embeddings = None

    # Layer graph
    input_ = lbann.Slice(
        lbann.Input(device='CPU'),
        slice_points=str_list([0, motif_size, motif_size+walk_length]),
        device='CPU',
    )
    motif_indices = lbann.Identity(input_, device='CPU')
    walk_indices = lbann.Identity(input_, device='CPU')
    gan = model.gan.CommunityGAN(
        num_vertices,
        motif_size,
        embed_dim,
        learn_rate,
        generator_type=generator_type,
        embeddings_device=embeddings_device,
        initial_embeddings=initial_embeddings,
    )
    loss, real_disc_prob, fake_disc_prob, gen_prob = gan(
        motif_indices,
        motif_size,
        walk_indices,
        walk_length,
    )

    # Metrics
    metrics = [
        lbann.Metric(real_disc_prob, name='D(real)'),
        lbann.Metric(fake_disc_prob, name='D(fake)'),
        lbann.Metric(gen_prob, name='G'),
    ]

    # Callbacks
    callbacks = [
        lbann.CallbackPrint(),
        lbann.CallbackTimer(),
        lbann.CallbackDumpWeights(directory=embeddings_dir,
                                  epoch_interval=1),
    ]
    if online_walker:
        callbacks.append(lbann.CallbackSetupCommunityGANDataReader())

    # Contruct model
    return lbann.Model(
        num_epochs,
        layers=lbann.traverse_layer_graph(input_),
        objective_function=loss,
        metrics=metrics,
        callbacks=callbacks,
    )

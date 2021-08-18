import lbann

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
):

    # Layer graph
    input_ = lbann.Slice(
        lbann.Input(data_field='samples'),
        slice_points=str_list([0, motif_size, motif_size+walk_length]),
    )
    motif_indices = lbann.Identity(input_)
    walk_indices = lbann.Identity(input_)
    gan = model.gan.CommunityGAN(
        num_vertices,
        motif_size,
        embed_dim,
        learn_rate,
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
                                  epoch_interval=num_epochs),
    ]

    # Perform computation at double precision
    for l in lbann.traverse_layer_graph(input_):
        l.datatype = lbann.DataType.DOUBLE
        for w in l.weights:
            w.datatype = lbann.DataType.DOUBLE

    # Contruct model
    return lbann.Model(
        num_epochs,
        layers=lbann.traverse_layer_graph(input_),
        objective_function=loss,
        metrics=metrics,
        callbacks=callbacks,
    )

import numpy as np
import lbann

import utils

def positive_samples_loss(
        sequence_length,
        encoder_embeddings,
        decoder_embeddings,
        scale_decay=0.8,
):

    # Compute similarity scores between encoder and decoder embeddings
    scores = lbann.MatMul(
        encoder_embeddings,
        decoder_embeddings,
        transpose_b=True,
    )
    scores = lbann.LogSigmoid(scores)

    # Scale similarity scores and add together
    # Note: The scaling factor decays exponentially as embeddings get
    # futher apart in the sequence.
    # Note: The sum of all the scaling factors is approximately -1.
    scale_dims = (sequence_length,sequence_length)
    scales = np.zeros(scale_dims)
    for i in range(sequence_length):
        for j in range(sequence_length):
            if i != j:
                scales[i,j] = (
                    -(1-scale_decay)/(2*scale_decay*sequence_length)
                    * scale_decay**np.abs(j-i)
                )
    scales = lbann.Weights(
        initializer=lbann.ValueInitializer(values=utils.str_list(np.nditer(scales))),
        optimizer=lbann.NoOptimizer(),
    )
    scales = lbann.WeightsLayer(dims=utils.str_list(scale_dims), weights=scales)
    loss = lbann.MatMul(
        lbann.Reshape(scores, dims='1 -1'),
        lbann.Reshape(scales, dims='1 -1'),
        transpose_b=True,
    )
    loss = lbann.Reshape(loss, dims='1')
    return loss

def negative_samples_loss(embeddings, negative_samples_embeddings):
    scores = lbann.MatMul(
        embeddings,
        negative_samples_embeddings,
        transpose_b=True,
    )
    scores = lbann.WeightedSum(scores, scaling_factors='-1')
    scores = lbann.LogSigmoid(scores)
    loss = lbann.Reduction(scores, mode='average')
    loss = lbann.WeightedSum(loss, scaling_factors='-1')
    return loss

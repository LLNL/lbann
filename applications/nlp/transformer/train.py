import math
import os.path

import lbann
import lbann.contrib.lc.launcher
from lbann.util import str_list

import dataset

# ----------------------------------------------
# Options
# ----------------------------------------------

# Dataset properties
vocab_size = dataset.vocab_size()
sequence_length = dataset.sequence_length
pad_index = dataset.pad_index

# ----------------------------------------------
# Layer graph
# ----------------------------------------------

def make_layer_graph(transformer, weights, params):

    # Input is two sequences of token IDs, separated by a pad token
    input_ = lbann.Identity(lbann.Input())

    # Get sequences of embedding vectors
    # Note: Scale embeddings by sqrt(embed_dim).
    # Note: Decoder input is shifted right, so first entry is pad token.
    embeddings_tokens = lbann.Identity(lbann.Slice(
        input_,
        axis=0,
        slice_points=str_list([0, 2*sequence_length]),
    ))
    embeddings = lbann.Embedding(
        embeddings_tokens,
        weights=weights['embedding'],
        num_embeddings=vocab_size,
        embedding_dim=params['embed_dim'],
        padding_idx=pad_index,
    )
    embeddings = lbann.WeightedSum(
        embeddings,
        scaling_factors=str(math.sqrt(params['embed_dim'])),
    )
    embeddings_slice = lbann.Slice(
        embeddings,
        axis=0,
        slice_points=str_list([0, sequence_length, 2*sequence_length]),
    )
    encoder_input = lbann.Identity(embeddings_slice)
    decoder_input = lbann.Identity(embeddings_slice)

    # Apply transformer model
    result = transformer(
        encoder_input, sequence_length,
        decoder_input, sequence_length,
    )

    # Use transformer decoder output to reconstruct decoder input
    # TODO: Use embedding weights
    preds = lbann.ChannelwiseFullyConnected(
        result,
        weights=[weights['classifier_matrix'], weights['classifier_bias']],
        output_channel_dims=[vocab_size],
    )
    preds = lbann.ChannelwiseSoftmax(preds)

    # Cross entropy loss with label smoothing
    label_tokens = lbann.Slice(
        input_,
        slice_points=str_list(range(sequence_length+1, 2*sequence_length+2)),
    )
    label_tokens = [lbann.Identity(label_tokens) for _ in range(sequence_length)]
    labels = [lbann.OneHot(token, size=vocab_size) for token in label_tokens]
    labels = lbann.Concatenation(
        [lbann.Reshape(label, dims=str_list([1, vocab_size])) for label in labels],
        axis=0,
    )
    if params['label_smoothing'] > 0:
        label_smoothing = params['label_smoothing']
        uniform_labels = lbann.Constant(
            value=1/vocab_size,
            num_neurons=str_list([sequence_length, vocab_size])
        )
        labels = lbann.WeightedSum(
            labels,
            uniform_labels,
            scaling_factors=str_list([1-label_smoothing, label_smoothing]),
        )
    loss = lbann.CrossEntropy(preds, labels)

    # Return layers and loss function
    return list(lbann.traverse_layer_graph(input_)), loss

# ----------------------------------------------
# Data reader
# ----------------------------------------------

def make_data_reader():
    reader = lbann.reader_pb2.DataReader()
    _reader = reader.reader.add()
    _reader.name = 'python'
    _reader.role = 'train'
    _reader.percent_of_data_to_use = 0.001421917
    #_reader.percent_of_data_to_use = 1.0
    _reader.python.module = 'dataset'
    _reader.python.module_dir = os.path.dirname(os.path.realpath(__file__))
    _reader.python.sample_function = 'get_train_sample'
    _reader.python.num_samples_function = 'num_train_samples'
    _reader.python.sample_dims_function = 'sample_dims'
    return reader

# ----------------------------------------------
# Batch script
# ----------------------------------------------

def make_batch_script(
    transformer,
    weights,
    work_dir,
    train_params,
    batch_params,
):

    # Create LBANN objects
    trainer = lbann.Trainer()
    reader = make_data_reader()
    layers, loss = make_layer_graph(transformer, weights, train_params)
    metrics = []
    callbacks = [
        lbann.CallbackPrint(),
        lbann.CallbackTimer(),
    ]
    model = lbann.Model(
        train_params['mini_batch_size'],
        train_params['num_epochs'],
        layers=layers,
        objective_function=loss,
        metrics=metrics,
        callbacks=callbacks
    )
    opt = lbann.Adam(learn_rate=0.0004, beta1=0.9, beta2=0.98, eps=1e-9) # TODO: LR schedule

    # Create Protobuf file
    protobuf_file = os.path.join(work_dir, 'experiment.prototext')
    lbann.proto.save_prototext(
        protobuf_file,
        trainer=trainer,
        model=model,
        data_reader=reader,
        optimizer=opt
    )

    # Create batch script
    script = lbann.contrib.lc.launcher.make_batch_script(
        work_dir=work_dir,
        **batch_params,
    )
    script.add_command('echo "Started training at $(date)"')
    script.add_parallel_command([
        lbann.lbann_exe(),
        f'--prototext={protobuf_file}',
    ])
    script.add_command('status=$?')
    script.add_command('echo "Finished training at $(date)"')
    script.add_command('exit ${status}')
    return script

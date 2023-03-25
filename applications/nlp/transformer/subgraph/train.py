"""Configure LBANN experiment with Transformer model."""
import math
import os.path

import lbann
import lbann.models
import lbann.contrib.launcher

import dataset

# ----------------------------------------------
# Options
# ----------------------------------------------

# Dataset properties
vocab_size = dataset.vocab_size()
sequence_length = dataset.sequence_length
pad_index = dataset.pad_index

# ----------------------------------------------
# Model
# ----------------------------------------------

def make_model(
    num_epochs,
    embed_dim,
    num_heads,
    label_smoothing,
    branches,
    subgraph_topology,
    num_encoder_layers,
    num_decoder_layers,
    filter_size,
    d_kv,
    subgraph_num_common_resources,
    ENABLE_ALLSUBGRAPH,
    ENABLE_Concat
):
    #branches = 4

    # Embedding weights
    var = 2 / (embed_dim + vocab_size) # Glorot initialization
    embedding_weights = lbann.Weights(
        name='embeddings',
        initializer=lbann.NormalInitializer(standard_deviation=math.sqrt(var)),
    )

    # Input is two sequences of token IDs
    input_ = lbann.Input(data_field='samples')

    # Get sequences of embedding vectors
    # Note: Scale embeddings by sqrt(embed_dim).
    # Note: Decoder input is shifted right, so embedding for last
    # token isn't needed.
    embeddings_tokens = lbann.Identity(lbann.Slice(
        input_,
        axis=0,
        slice_points=[0, 2*sequence_length-1],
    ))
    embeddings = lbann.Embedding(
        embeddings_tokens,
        weights=embedding_weights,
        num_embeddings=vocab_size,
        embedding_dim=embed_dim,
        padding_idx=pad_index,
    )
    embeddings = lbann.WeightedSum(
        embeddings,
        scaling_factors=math.sqrt(embed_dim),
    )
    embeddings_slice = lbann.Slice(
        embeddings,
        axis=0,
        slice_points=[0, sequence_length, 2*sequence_length-1],
    )
    encoder_input = lbann.Identity(embeddings_slice)
    decoder_input = lbann.Identity(embeddings_slice)

    # Apply transformer model
    transformer = lbann.models.subgraph.TransformerSubGraph(branches = branches,
        hidden_size=embed_dim,
        num_heads=num_heads,
        num_encoder_layers = num_encoder_layers,
        num_decoder_layers = num_decoder_layers,
        filter_size = filter_size,
        d_kv = d_kv,
        name='transformer',
        ENABLE_ALLSUBGRAPH = ENABLE_ALLSUBGRAPH,
        ENABLE_Concat = ENABLE_Concat
    )
    result = transformer(
        encoder_input, sequence_length,
        decoder_input, sequence_length-1,
    )

    # Reconstruct decoder input
    preds = lbann.ChannelwiseFullyConnected(
        result,
        weights=embedding_weights,
        output_channel_dims=[vocab_size],
        bias=False,
        transpose=True,
    )
    preds = lbann.ChannelwiseSoftmax(preds)
    preds = lbann.Slice(preds, axis=0, slice_points=range(sequence_length))
    preds = [lbann.Identity(preds) for _ in range(sequence_length-1)]

    # Count number of non-pad tokens
    label_tokens = lbann.Identity(lbann.Slice(
        input_,
        slice_points=[sequence_length+1, 2*sequence_length],
    ))
    pads = lbann.Constant(value=pad_index, num_neurons=sequence_length-1)
    is_not_pad = lbann.NotEqual(label_tokens, pads)
    num_not_pad = lbann.Reduction(is_not_pad, mode='sum')

    # Cross entropy loss with label smoothing
    label_tokens = lbann.Slice(
        label_tokens,
        slice_points=range(sequence_length),
    )
    label_tokens = [lbann.Identity(label_tokens) for _ in range(sequence_length-1)]
    if label_smoothing > 0:
        uniform_label = lbann.Constant(
            value=1/vocab_size,
            num_neurons=[1, vocab_size]
        )
    loss = []
    for i in range(sequence_length-1):
        label = lbann.OneHot(label_tokens[i], size=vocab_size)
        label = lbann.Reshape(label, dims=[1, vocab_size])
        if label_smoothing > 0:
            label = lbann.WeightedSum(
                label,
                uniform_label,
                scaling_factors=[1-label_smoothing, label_smoothing],
            )
        loss.append(lbann.CrossEntropy(preds[i], label))
    loss = lbann.Concatenation(loss)

    # Average cross entropy over non-pad tokens
    loss_scales = lbann.Divide(
        is_not_pad,
        lbann.Tessellate(num_not_pad, hint_layer=is_not_pad),
    )
    loss = lbann.Multiply(loss, loss_scales)
    loss = lbann.Reduction(loss, mode='sum')

    # Construct model
    metrics = []
    callbacks = [lbann.CallbackPrint(), lbann.CallbackTimer()]


    return lbann.Model(
        num_epochs,
        subgraph_communication=lbann.SubgraphCommunication.COLL_OPT,
        subgraph_topology=subgraph_topology,
        subgraph_num_common_resources = subgraph_num_common_resources,
        layers=lbann.traverse_layer_graph(input_),
        objective_function=loss,
        metrics=metrics,
        callbacks=callbacks,
    )

# ----------------------------------------------
# Data reader
# ----------------------------------------------

def make_data_reader():
    reader = lbann.reader_pb2.DataReader()
    _reader = reader.reader.add()
    _reader.name = 'python'
    _reader.role = 'train'
    _reader.shuffle = True
    _reader.percent_of_data_to_use = 1.0
    _reader.python.module = 'dataset'
    _reader.python.module_dir = os.path.dirname(os.path.realpath(__file__))
    _reader.python.sample_function = 'get_train_sample'
    _reader.python.num_samples_function = 'num_train_samples'
    _reader.python.sample_dims_function = 'sample_dims'
    return reader

# ----------------------------------------------
# Batch script
# ----------------------------------------------

def make_batch_script(trainer_params, model_params, script_params):

    # Create LBANN objects
    trainer = lbann.Trainer(mini_batch_size=trainer_params['mini_batch_size'])
    model = make_model(**model_params)
    reader = make_data_reader()

    # Optimizer with learning rate schedule
    # Note: Rough approximation of
    #   embed_dim^-0.5 * min(step^-0.5, step*warmup^-1.5)
    # with embed_dim=512 and warmup=4000.
    opt = lbann.Adam(learn_rate=0.0001, beta1=0.9, beta2=0.98, eps=1e-9)
    # opt = lbann.NoOptimizer()
    model.callbacks.append(
        lbann.CallbackDropFixedLearningRate(
            drop_epoch=[1],
            amt=2,
        )
    )
    model.callbacks.append(
        lbann.CallbackDropFixedLearningRate(
            drop_epoch=[2,4,8,12],
            amt=0.75,
        )
    )

    # Checkpoint after every epoch
    # trainer.callbacks.append(
    #     lbann.CallbackCheckpoint(
    #         checkpoint_dir=os.path.join(script_params['work_dir'], 'checkpoint'),
    #         checkpoint_epochs=1,
    #     )
    # )

    # Dump weights after every epoch
    # model.callbacks.append(
    #     lbann.CallbackDumpWeights(
    #         basename=os.path.join(script_params['work_dir'], 'weights'),
    #         epoch_interval=1,
    #     )
    # )

    # Create Protobuf file
    protobuf_file = os.path.join(script_params['work_dir'], 'experiment.prototext')
    lbann.proto.save_prototext(
        protobuf_file,
        trainer=trainer,
        model=model,
        data_reader=reader,
        optimizer=opt,
    )

    # Create batch script
    script = lbann.contrib.launcher.make_batch_script(
        **script_params,
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

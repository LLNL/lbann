"""Configure LBANN experiment with Transformer model."""
import math
import os.path

import lbann
import lbann.models
import lbann.contrib.launcher



DAMPING_PARAM_NAMES = ["act", "err", "bn_act", "bn_err"]


# ----------------------------------------------
# Model
# ----------------------------------------------

def make_model(
    num_epochs,
    embed_dim,
    num_heads,
    num_layers,
    vocab_size,
    sequence_length,
    pad_index,
    dropout
):

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
    transformer = lbann.models.Transformer(
        hidden_size=embed_dim,
        num_heads=num_heads,
        dropout=dropout,
        name='transformer',
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers
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
        name="prediction_layer"
    )
    preds = lbann.ChannelwiseSoftmax(preds)
    preds = lbann.TensorPermute(preds, axes=[1, 0])

    # Compute labels
    label_tokens = lbann.Identity(lbann.Slice(
        input_,
        slice_points=[sequence_length+1, 2*sequence_length],
    ))
    labels = lbann.Reshape(label_tokens, dims=[1, sequence_length - 1])

    # Filter out output predictions that are in padding from cross-entropy by
    # using values that will never contribute to the cross-entropy loss
    labels = lbann.Select(labels, lbann.Identity(labels), value=pad_index, if_false=(vocab_size + 1))

    # Compute cross-entropy
    loss = lbann.CrossEntropy(preds, labels, use_labels=True)

    # Construct model
    metrics = []
    callbacks = [lbann.CallbackPrint(), lbann.CallbackTimer()]
    return lbann.Model(
        num_epochs,
        layers=lbann.traverse_layer_graph(input_),
        objective_function=loss,
        metrics=metrics,
        callbacks=callbacks,
    )

# ----------------------------------------------
# Data reader
# ----------------------------------------------

def make_data_reader(synthetic, fraction):
    reader = lbann.reader_pb2.DataReader()
    _reader = reader.reader.add()
    _reader.name = 'python'
    _reader.role = 'train'
    _reader.shuffle = True
    _reader.fraction_of_data_to_use = fraction
    _reader.python.module = 'dataset_synthetic' if synthetic else 'dataset'
    _reader.python.module_dir = os.path.dirname(os.path.realpath(__file__))
    _reader.python.sample_function = 'get_train_sample'
    _reader.python.num_samples_function = 'num_train_samples'
    _reader.python.sample_dims_function = 'sample_dims'
    return reader

# ----------------------------------------------
# Batch script
# ----------------------------------------------

def make_batch_script(trainer_params, model_params, script_params, args):
    algo = lbann.BatchedIterativeOptimizer("sgd", epoch_count=args.num_epochs)
    if args.kfac:
        kfac_args = {}
        if args.kfac_use_pi:
            kfac_args["use_pi"] = 1
        if args.print_matrix:
            kfac_args["print_matrix"] = 1
        if args.print_matrix_summary:
            kfac_args["print_matrix_summary"] = 1
        for n in DAMPING_PARAM_NAMES:
            kfac_args["damping_{}".format(n)] = getattr(
                args, "kfac_damping_{}".format(n)).replace(",", " ")
        if args.kfac_damping_warmup_steps > 0:
            kfac_args["damping_warmup_steps"] = args.kfac_damping_warmup_steps
        if args.kfac_update_interval_init != 1 or args.kfac_update_interval_target != 1:
            kfac_args["update_intervals"] = "{} {}".format(
                args.kfac_update_interval_init,
                args.kfac_update_interval_target,
            )
        if args.kfac_update_interval_steps != 1:
            kfac_args["update_interval_steps"] = args.kfac_update_interval_steps
        kfac_args["kronecker_decay"] = 0.95
        kfac_args["compute_interval"] = args.kfac_compute_interval_steps
        kfac_args["distribute_precondition_compute"] = args.enable_distribute_compute
        kfac_args["disable_layers"]="prediction_layer"
        kfac_args["use_eigen_decomposition"] = args.use_eigen
        kfac_args["kfac_use_interval"] = args.kfac_sgd_mix

        algo = lbann.KFAC("kfac", algo, **kfac_args)

    # Create LBANN objects
    trainer = lbann.Trainer(mini_batch_size=trainer_params['mini_batch_size'], training_algo=algo)

    # Import dataset
    if args.synthetic:
        import dataset_synthetic as dataset
    else:
        import dataset

    # Dataset properties
    vocab_size = dataset.vocab_size()
    sequence_length = dataset.sequence_length
    pad_index = dataset.pad_index

    model = make_model(vocab_size=vocab_size, sequence_length=sequence_length, pad_index=pad_index, **model_params)
    reader = make_data_reader(args.synthetic, args.dataset_fraction)

    # Optimizer with learning rate schedule
    # Note: Rough approximation of
    #   embed_dim^-0.5 * min(step^-0.5, step*warmup^-1.5)
    # with embed_dim=512 and warmup=4000.
    opt = lbann.Adam(learn_rate=0.0001, beta1=0.9, beta2=0.98, eps=1e-9)
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

    if args.checkpoint:
        # Checkpoint after every epoch
        trainer.callbacks.append(
            lbann.CallbackCheckpoint(
                checkpoint_dir=os.path.join(script_params['work_dir'], 'checkpoint'),
                checkpoint_epochs=1,
            )
        )

        # Dump weights after every epoch
        model.callbacks.append(
            lbann.CallbackDumpWeights(
                directory=os.path.join(script_params['work_dir'], 'weights'),
                epoch_interval=1,
            )
        )

    # Print a progress bar
    if args.progress:
        model.callbacks.append(
            lbann.CallbackProgressBar()
        )

    profiler = create_profile_callback(args)
    if profiler is not None:
        model.callbacks.append(profiler)

    kwargs = lbann.contrib.args.get_scheduler_kwargs(args)

    # lbann.contrib.launcher.run(trainer, model, reader, opt,
    #                        job_name="transformer",
    #                        environment = {
    #                           'LBANN_USE_CUBLAS_TENSOR_OPS' : 0,
    #                           'LBANN_USE_CUDNN_TENSOR_OPS' : 0,
    #                           "LBANN_KEEP_ERROR_SIGNALS": 1
    #                       },
    #                       lbann_args=" ", **kwargs)

    script_params['environment'] =                     {       
                              'LBANN_USE_CUBLAS_TENSOR_OPS' : 0,
                              'LBANN_USE_CUDNN_TENSOR_OPS' : 0,
                              "LBANN_KEEP_ERROR_SIGNALS": 1
                          }

    # Create Protobuf file
    protobuf_file = os.path.join(script_params['work_dir'], 'experiment.prototext')

    lbann.proto.save_prototext(
        protobuf_file,
        trainer=trainer,
        model=model,
        data_reader=reader,
        optimizer=opt
    )


    # # Create batch script
    script = lbann.contrib.launcher.make_batch_script(
        **script_params,
    )
    script.add_command('echo "Started training at $(date)"')
    script.add_parallel_command([
        lbann.lbann_exe(),
        f'--prototext={protobuf_file}',
    ] + lbann.contrib.args.get_profile_args(args))
    script.add_command('status=$?')
    script.add_command('echo "Finished training at $(date)"')
    script.add_command('exit ${status}')
    return script

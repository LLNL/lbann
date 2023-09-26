import argparse
from enum import Enum, auto
import lbann
import math
from typing import Tuple
from lbann.models.transformer import LayerNorm
from lbann.modules.transformer import PositionalEncoding, LearnedInputEncoding
import numpy as np


class InputEncoding(Enum):
    """ Different types of input encoding used by the transformer samples. """
    POSITIONAL = auto()  # Positional encoding
    LEARNED = auto()  # Learned embeddings
    NONE = auto()  # No encoding


def create_encoder_decoder_transformer(dataset, args: argparse.Namespace):
    """
    Creates an Encoder-Decoder Transformer model as per Vaswani et al., 
    "Attention is all you need" (2017), with a language modeling head for
    sequence transduction tasks (e.g. translation).
    """
    num_encoders = num_decoders = args.num_layers
    sequence_length = dataset.sequence_length
    vocab_size = dataset.vocab_size()

    # Embedding weights
    var = 2 / (args.embed_dim + vocab_size)  # Glorot initialization
    embedding_weights = lbann.Weights(
        name='embeddings',
        initializer=lbann.NormalInitializer(standard_deviation=math.sqrt(var)),
    )

    # Input is two sequences of token IDs
    input_tokens = lbann.Input(data_field='samples')

    # Get sequences of embedding vectors
    # Note: Scale embeddings by sqrt(embed_dim).
    # Note: Decoder input is shifted right, so embedding for last
    # token isn't needed.
    embeddings_tokens = lbann.Identity(
        lbann.Slice(
            input_tokens,
            axis=0,
            slice_points=[0, 2 * sequence_length - 1],
        ))
    embeddings = lbann.Embedding(
        embeddings_tokens,
        weights=embedding_weights,
        num_embeddings=vocab_size,
        embedding_dim=args.embed_dim,
        padding_idx=dataset.pad_index,
    )
    embeddings = lbann.WeightedSum(
        embeddings,
        scaling_factors=math.sqrt(args.embed_dim),
    )
    embeddings_slice = lbann.Slice(
        embeddings,
        axis=0,
        slice_points=[0, sequence_length, 2 * sequence_length - 1],
    )
    encoder_input = lbann.Identity(embeddings_slice)
    decoder_input = lbann.Identity(embeddings_slice)

    # Apply input encoding
    encoder_input, decoder_input = _add_input_encoding(
        encoder_input, decoder_input, InputEncoding.POSITIONAL, args.embed_dim,
        args.input_dropout, sequence_length, sequence_length - 1)

    # Add encoder-decoder transformer model
    transformer = lbann.models.Transformer(
        hidden_size=args.embed_dim,
        num_heads=args.num_attention_heads,
        dropout=args.dropout,
        feedforward_size=args.feedforward_dim,
        name='transformer',
        num_encoder_layers=num_encoders,
        num_decoder_layers=num_decoders)

    # Run through transformer
    result = transformer(encoder_input, decoder_input, sequence_length - 1)

    # Reconstruct decoder input
    preds = lbann.ChannelwiseFullyConnected(result,
                                            weights=embedding_weights,
                                            output_channel_dims=[vocab_size],
                                            bias=False,
                                            transpose=True,
                                            name="prediction_layer")
    preds = lbann.ChannelwiseSoftmax(preds)
    preds = lbann.TensorPermute(preds, axes=[1, 0])

    # Compute loss
    loss = _add_encoder_decoder_loss(preds, input_tokens, sequence_length,
                                     vocab_size, dataset.pad_index)

    # Construct model
    metrics = []
    callbacks = [lbann.CallbackPrint(), lbann.CallbackTimer()]
    return lbann.Model(
        args.num_epochs,
        layers=lbann.traverse_layer_graph(input_tokens),
        objective_function=loss,
        metrics=metrics,
        callbacks=callbacks,
    )


def create_causal_lm_decoder_transformer(dataset, embed_dim: int,
                                         num_decoders: int, num_heads: int,
                                         dropout: float, input_dropout: float,
                                         attn_dropout: float, num_epochs: int):
    """
    Creates a GPT-style decoder-only transformer for causal language modeling
    tasks (e.g., predict next token).
    """
    sequence_length = dataset.sequence_length
    vocab_size = dataset.vocab_size()

    # Embedding weights
    var = 2 / (embed_dim + vocab_size)  # Glorot initialization
    embedding_weights = lbann.Weights(
        name='embeddings',
        initializer=lbann.NormalInitializer(standard_deviation=math.sqrt(var)),
    )

    # Input is a sequences of token IDs
    input_tokens = lbann.Input(data_field='samples')

    # Get sequences of embedding vectors
    embeddings = lbann.Embedding(
        input_tokens,
        weights=embedding_weights,
        num_embeddings=vocab_size,
        embedding_dim=embed_dim,
        padding_idx=dataset.pad_index,
    )
    decoder_input = lbann.WeightedSum(
        embeddings,
        scaling_factors=math.sqrt(embed_dim),
    )

    # Apply input encoding
    _, decoder_input = _add_input_encoding(None, decoder_input,
                                           InputEncoding.LEARNED, embed_dim,
                                           input_dropout, 0, sequence_length)

    # Add a GPT-style (decoder-only) transformer model
    transformer = lbann.models.Transformer(hidden_size=embed_dim,
                                           num_heads=num_heads,
                                           dropout=dropout,
                                           attn_dropout=attn_dropout,
                                           num_encoder_layers=0,
                                           num_decoder_layers=num_decoders,
                                           pre_layernorm=True,
                                           activation=lbann.Gelu,
                                           name='transformer')

    # Run through transformer with the same sequence
    result = transformer(decoder_input, decoder_input, sequence_length)

    # Apply layer normalization on the outputs
    norm_final = LayerNorm(embed_dim, name=f'final_layernorm')
    result = norm_final(result)

    # Apply language modeling head on results
    preds = lbann.ChannelwiseFullyConnected(result,
                                            weights=embedding_weights,
                                            output_channel_dims=[vocab_size],
                                            bias=False,
                                            transpose=True,
                                            name="prediction_layer")
    preds = lbann.ChannelwiseSoftmax(preds)
    preds = lbann.TensorPermute(preds, axes=[1, 0])

    # Compute loss
    loss = _add_autoregressive_loss(preds, input_tokens, sequence_length,
                                    vocab_size, dataset.pad_index)

    # Construct model
    metrics = []
    callbacks = [lbann.CallbackPrint(), lbann.CallbackTimer()]
    return lbann.Model(
        num_epochs,
        layers=lbann.traverse_layer_graph(input_tokens),
        objective_function=loss,
        metrics=metrics,
        callbacks=callbacks,
    )


def _add_input_encoding(
        encoder_input: lbann.Layer, decoder_input: lbann.Layer,
        encoding_kind: InputEncoding, embed_dim: int, input_dropout: float,
        encoder_sequence_length: int,
        decoder_sequence_length: int) -> Tuple[lbann.Layer, lbann.Layer]:
    if encoding_kind == InputEncoding.NONE:
        # Do nothing
        return encoder_input, decoder_input

    elif encoding_kind == InputEncoding.POSITIONAL:
        # Trigonometric positional encoding
        positional_encoder = PositionalEncoding(embed_dim, input_dropout)
        kwargs = {}
    elif encoding_kind == InputEncoding.LEARNED:
        # Learned (embedding) encoding
        max_seqlen = max(encoder_sequence_length, decoder_sequence_length)
        positional_encoder = LearnedInputEncoding(embed_dim, max_seqlen,
                                                  input_dropout)
        # Optimize by not computing embeddings twice
        kwargs = dict(learned_encoding=positional_encoder.compute_embeddings())

    # Apply encoder
    if encoder_input is not None:
        encoder_input = positional_encoder(encoder_input,
                                           encoder_sequence_length, **kwargs)
    if decoder_input is not None:
        decoder_input = positional_encoder(decoder_input,
                                           decoder_sequence_length, **kwargs)

    return encoder_input, decoder_input


def _add_encoder_decoder_loss(preds, both_sequences, sequence_length,
                              vocab_size, pad_index):
    # Get label tokens from the second sequence, starting from the second token
    # onwards
    target_sequence = lbann.Identity(
        lbann.Slice(
            both_sequences,
            slice_points=[sequence_length + 1, 2 * sequence_length],
        ))
    labels = lbann.Reshape(target_sequence, dims=[1, sequence_length - 1])

    # Filter out output predictions that are in padding from cross-entropy by
    # using values that will never contribute to the cross-entropy loss
    labels = lbann.Select(labels,
                          lbann.Identity(labels),
                          value=pad_index,
                          if_false=(vocab_size + 1))

    # Compute cross-entropy
    return lbann.CrossEntropy(preds, labels, use_labels=True)


def _add_autoregressive_loss(preds, input_tokens, sequence_length, vocab_size,
                             pad_index):
    # Compute cross-entropy loss between preds[:-1] (up until the last token)
    # and input[1:] (predicting one token forward)
    shifted_preds = lbann.Identity(
        lbann.Slice(preds, axis=1, slice_points=[0, sequence_length - 1]))
    shifted_labels = lbann.Identity(
        lbann.Slice(input_tokens, slice_points=[1, sequence_length]))
    flat_labels = lbann.Reshape(shifted_labels, dims=[1, sequence_length - 1])

    # Flatten labels
    return lbann.CrossEntropy(shifted_preds, flat_labels, use_labels=True)


# Command-line arguments
def add_transformer_architecture_arguments(args: argparse.Namespace):
    """
    Adds the command line arguments to specify transformer architecture model
    parameters. This is only relevant for the encoder-decoder transformer model.
    """
    args.add_argument('--num-attention-heads',
                      action='store',
                      default=8,
                      type=int,
                      help='number of parallel attention layers (default: 8)',
                      metavar='NUM')
    args.add_argument('--embed-dim',
                      action='store',
                      default=512,
                      type=int,
                      help='embedding space dimension (default: 512)',
                      metavar='NUM')
    args.add_argument('--feedforward-dim',
                      action='store',
                      default=0,
                      type=int,
                      help='feedforward network dimension. If zero, set to be '
                      '4 times the embedding dimension (default: 0)',
                      metavar='NUM')
    args.add_argument('--num-layers',
                      action='store',
                      default=6,
                      type=int,
                      help='Number of encoder and decoder layers (default: 6)',
                      metavar='NUM')

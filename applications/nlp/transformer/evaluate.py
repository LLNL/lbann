"""Evaluate Transformer example.

The LBANN Transformer model is assumed to have saved its weights to
weight files with the "dump weights" callback. These weights are
loaded into a PyTorch model and English-German translation is
performed with greedy decoding on the WMT 2014 validation dataset.
BLEU scores are computed for the predicted translations.

"""

import argparse
import os.path
import sys

import numpy as np
import torch
import torch.nn
import torchnlp.metrics

# Local imports
current_file = os.path.realpath(__file__)
root_dir = os.path.dirname(os.path.dirname(current_file))
sys.path.append(root_dir)
import dataset
import utils
import utils.paths

# ----------------------------------------------
# Options
# ----------------------------------------------

# Evaluation options
mini_batch_size = 64    # Doesn't need to match training

# Hard-coded model parameters
# Note: Must match parameters from training.
embed_dim = 512
num_heads = 8
num_encoder_layers = 6
num_decoder_layers = 6
filter_dim = 2048
dropout = 0.1

# Dataset properties
vocab_size = dataset.vocab_size()
max_sequence_length = dataset.sequence_length
bos_index = dataset.bos_index
eos_index = dataset.eos_index
pad_index = dataset.pad_index
num_samples = dataset.num_val_samples()

# ----------------------------------------------
# Evaluation data
# ----------------------------------------------

def get_batch(indices):
    """Get a batch of samples from the evaluation dataset.

    The sequences are padded to the length of the longest sequence in
    the batch.

    """

    # Get data samples
    indices = utils.make_iterable(indices)
    tokens_list_en = []
    tokens_list_de = []
    for index in indices:
        tokens_en, tokens_de = dataset.get_val_sample(index)
        tokens_list_en.append(tokens_en)
        tokens_list_de.append(tokens_de)

    # Convert tokens to PyTorch tensors
    tokens_en = np.full(
        (max(len(seq) for seq in tokens_list_en), len(indices)),
        pad_index,
        dtype=int,
    )
    tokens_de = np.full(
        (max(len(seq) for seq in tokens_list_de), len(indices)),
        pad_index,
        dtype=int,
    )
    for i, seq in enumerate(tokens_list_en):
        tokens_en[:len(seq), i] = seq
    for i, seq in enumerate(tokens_list_de):
        tokens_de[:len(seq), i] = seq
    tokens_en = torch.from_numpy(tokens_en)
    tokens_de = torch.from_numpy(tokens_de)
    return tokens_en, tokens_de

# ----------------------------------------------
# Load model from file
# ----------------------------------------------

def load_parameter(weight_file):
    """Create a PyTorch Parameter object with weights from LBANN.

    Weight file is assumed to have been created by the "dump weights"
    callback in LBANN.

    """
    data = np.loadtxt(weight_file, dtype=np.float32)
    return torch.nn.Parameter(
        data=torch.from_numpy(data),
        requires_grad=False)

def load_embedding_layer(weights_prefix):
    """Create a PyTorch embedding layer with weights from LBANN.

    Weight files are assumed to have been created by the "dump
    weights" callback in LBANN. They should be in the form
    <weights_prefix>-embeddings-Weights.txt.

    """
    weight_file = f'{weights_prefix}/embeddings.txt'
    weight = load_parameter(weight_file).transpose(1,0)
    return torch.nn.Embedding(
        num_embeddings=vocab_size,
        embedding_dim=embed_dim,
        padding_idx=pad_index,
        _weight=weight,
    )

def load_transformer(weights_prefix):
    """Create a PyTorch transformer with weights from LBANN.

    Weight files are assumed to have been created by the "dump
    weights" callback in LBANN. They should be in the form
    <weights_prefix>-<weight_name>-Weights.txt.

    """

    # PyTorch transformer model
    transformer = torch.nn.Transformer(
        d_model=embed_dim,
        nhead=num_heads,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=filter_dim,
        dropout=dropout,
    )

    # Set transformer to evaluation mode
    transformer.eval()

    # Load weights for encoder
    for i, layer in enumerate(transformer.encoder.layers):

        # Load weights for self-attention
        attention = layer.self_attn
        attention._qkv_same_embed_dim = False
        prefix = f'{weights_prefix}/transformer_encoder{i}_attention'
        attention.q_proj_weight = load_parameter(f'{prefix}_query_matrix.txt')
        attention.q_proj_bias = load_parameter(f'{prefix}_query_bias.txt')
        attention.k_proj_weight = load_parameter(f'{prefix}_key_matrix.txt')
        attention.k_proj_bias = load_parameter(f'{prefix}_key_bias.txt')
        attention.v_proj_weight = load_parameter(f'{prefix}_value_matrix.txt')
        attention.v_proj_bias = load_parameter(f'{prefix}_value_bias.txt')
        attention.out_proj_weight = load_parameter(f'{prefix}_output_matrix.txt')
        attention.out_proj_bias = load_parameter(f'{prefix}_output_bias.txt')

        # Load weights for feedforward network
        prefix = f'{weights_prefix}/transformer_encoder{i}'
        layer.linear1.weight = load_parameter(f'{prefix}_fc1_matrix.txt')
        layer.linear1.bias = load_parameter(f'{prefix}_fc1_bias.txt')
        layer.linear2.weight = load_parameter(f'{prefix}_fc2_matrix.txt')
        layer.linear2.bias = load_parameter(f'{prefix}_fc2_bias.txt')
        layer.norm1.weight = load_parameter(f'{prefix}_norm1_weight.txt')
        layer.norm1.bias = load_parameter(f'{prefix}_norm1_bias.txt')
        layer.norm2.weight = load_parameter(f'{prefix}_norm2_weight.txt')
        layer.norm2.bias = load_parameter(f'{prefix}_norm2_bias.txt')

    # Load weights for decoder
    for i, layer in enumerate(transformer.decoder.layers):

        # Load weights for self-attention
        attention = layer.self_attn
        attention._qkv_same_embed_dim = False
        prefix = f'{weights_prefix}/transformer_decoder{i}_attention1'
        attention.q_proj_weight = load_parameter(f'{prefix}_query_matrix.txt')
        attention.q_proj_bias = load_parameter(f'{prefix}_query_bias.txt')
        attention.k_proj_weight = load_parameter(f'{prefix}_key_matrix.txt')
        attention.k_proj_bias = load_parameter(f'{prefix}_key_bias.txt')
        attention.v_proj_weight = load_parameter(f'{prefix}_value_matrix.txt')
        attention.v_proj_bias = load_parameter(f'{prefix}_value_bias.txt')
        attention.out_proj_weight = load_parameter(f'{prefix}_output_matrix.txt')
        attention.out_proj_bias = load_parameter(f'{prefix}_output_bias.txt')

        # Load weights for attention with memory
        attention = layer.multihead_attn
        attention._qkv_same_embed_dim = False
        prefix = f'{weights_prefix}/transformer_decoder{i}_attention2'
        attention.q_proj_weight = load_parameter(f'{prefix}_query_matrix.txt')
        attention.q_proj_bias = load_parameter(f'{prefix}_query_bias.txt')
        attention.k_proj_weight = load_parameter(f'{prefix}_key_matrix.txt')
        attention.k_proj_bias = load_parameter(f'{prefix}_key_bias.txt')
        attention.v_proj_weight = load_parameter(f'{prefix}_value_matrix.txt')
        attention.v_proj_bias = load_parameter(f'{prefix}_value_bias.txt')
        attention.out_proj_weight = load_parameter(f'{prefix}_output_matrix.txt')
        attention.out_proj_bias = load_parameter(f'{prefix}_output_bias.txt')

        # Load weights for feedforward network
        prefix = f'{weights_prefix}/transformer_decoder{i}'
        layer.linear1.weight = load_parameter(f'{prefix}_fc1_matrix.txt')
        layer.linear1.bias = load_parameter(f'{prefix}_fc1_bias.txt')
        layer.linear2.weight = load_parameter(f'{prefix}_fc2_matrix.txt')
        layer.linear2.bias = load_parameter(f'{prefix}_fc2_bias.txt')
        layer.norm1.weight = load_parameter(f'{prefix}_norm1_weight.txt')
        layer.norm1.bias = load_parameter(f'{prefix}_norm1_bias.txt')
        layer.norm2.weight = load_parameter(f'{prefix}_norm2_weight.txt')
        layer.norm2.bias = load_parameter(f'{prefix}_norm2_bias.txt')
        layer.norm3.weight = load_parameter(f'{prefix}_norm3_weight.txt')
        layer.norm3.bias = load_parameter(f'{prefix}_norm3_bias.txt')

    return transformer

# ----------------------------------------------
# Evaluate transformer model
# ----------------------------------------------

def add_positional_encoding(x):
    """Add positional encoding for transformer model."""
    sequence_length = x.shape[0]
    embed_dim = x.shape[2]
    encoding = np.zeros(x.shape, dtype=np.float32)
    for i in range((embed_dim+1) // 2):
        pos = np.arange(sequence_length).reshape(-1,1)
        encoding[:,:,2*i] = np.sin(pos / 10000**(2*i/embed_dim))
    for i in range(embed_dim // 2):
        pos = np.arange(sequence_length).reshape(-1,1)
        encoding[:,:,2*i+1] = np.cos(pos / 10000**(2*i/embed_dim))
    return x + torch.from_numpy(encoding)

def greedy_decode(tokens_en, embedding_layer, transformer, classifier):
    """Generate sequence with transformer.

    Predict tokens one at a time by choosing the one that maximizes
    the classification score.

    """

    # Encode English sequence
    embeddings_en = embedding_layer(tokens_en)
    memory = transformer.encoder(
        add_positional_encoding(embeddings_en * np.sqrt(embed_dim))
    )

    # Decode German sequence
    # TODO: Only perform compute for last sequence entry
    # TODO: Detect EOS tokens and stop early
    tokens_de = torch.full((1,tokens_en.shape[1]), bos_index, dtype=int)
    for i in range(1, max_sequence_length):
        embeddings_de = embedding_layer(tokens_de)
        preds = transformer.decoder(
            add_positional_encoding(embeddings_de * np.sqrt(embed_dim)),
            memory,
            tgt_mask=transformer.generate_square_subsequent_mask(i),
        )
        preds = classifier(preds[-1,:,:])
        preds = preds.argmax(dim=1)
        tokens_de = torch.cat([tokens_de, preds.reshape(1,-1)], dim=0)
    return tokens_de

def evaluate_transformer(weights_prefix):
    """Evaluate transformer model with weights from LBANN.

    Weight files are assumed to have been created by the "dump
    weights" callback in LBANN. They should be in the form
    <weights_prefix>-<weight_name>-Weights.txt.

    """

    # Load model weights from file
    embedding_layer = load_embedding_layer(weights_prefix)
    transformer = load_transformer(weights_prefix)
    classifier = torch.nn.Linear(embed_dim, vocab_size, bias=False)
    classifier.weight = embedding_layer.weight

    # Evaluate model
    bleu_scores = []
    for batch, index_start in enumerate(range(0, num_samples, mini_batch_size)):
        index_end = min(index_start+mini_batch_size, num_samples)
        indices = list(range(index_start, index_end))
        batch_size = len(indices)

        # Translate English sequence to German
        # TODO: Decoding with beam search
        tokens_en, true_tokens_de = get_batch(indices)
        pred_tokens_de = greedy_decode(
            tokens_en,
            embedding_layer,
            transformer,
            classifier,
        )

        # Compute BLEU score
        for i in range(batch_size):
            hypothesis = dataset.detokenize(pred_tokens_de[:,i].numpy())
            reference = dataset.detokenize(true_tokens_de[:,i].numpy())
            bleu_scores.append(
                torchnlp.metrics.get_moses_multi_bleu(
                    [hypothesis],
                    [reference],
                )
            )

    # Print results
    print(
        f'BLEU score: '
        f'mean={np.mean(bleu_scores)}, '
        f'stdev={np.std(bleu_scores)}, '
        f'min={np.min(bleu_scores)}, '
        f'max={np.max(bleu_scores)}'
    )

# ----------------------------------------------
# Command-line options if run as script
# ----------------------------------------------

if __name__ == "__main__":

    # Command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'weights_prefix', type=str,
        help='prefix for saved weights from LBANN')
    args = parser.parse_args()

    # Evaluate model
    evaluate_transformer(args.weights_prefix)

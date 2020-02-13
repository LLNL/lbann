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

# Hard-coded
mini_batch_size = 64
embed_dim = 512
num_heads = 8
num_encoder_layers = 6
num_decoder_layers = 6
filter_dim = 2048
dropout = 0.1

# Dataset properties
vocab_size = dataset.vocab_size()
sequence_length = dataset.sequence_length
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
    """Create a PyTorch Parameter object with weights from LBANN."""
    data = np.loadtxt(weight_file, dtype=np.float32)
    return torch.nn.Parameter(
        data=torch.from_numpy(data),
        requires_grad=False
    )

def load_embedding_layer(weights_prefix):
    """Create a PyTorch embedding layer with weights from LBANN."""
    weight_file = f'{weights_prefix}-embeddings-Weights.txt'
    weight = load_parameter(weight_file).transpose(1,0)
    return torch.nn.Embedding(
        num_embeddings=vocab_size,
        embedding_dim=embed_dim,
        padding_idx=pad_index,
        _weight=weight,
    )

def load_transformer(weights_prefix):
    """Create a PyTorch transformer with weights from LBANN."""

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
        prefix = f'{weights_prefix}-transformer_encoder{i}_attention'
        attention.q_proj_weight = load_parameter(f'{prefix}_query_matrix-Weights.txt')
        attention.q_proj_bias = load_parameter(f'{prefix}_query_bias-Weights.txt')
        attention.k_proj_weight = load_parameter(f'{prefix}_key_matrix-Weights.txt')
        attention.k_proj_bias = load_parameter(f'{prefix}_key_bias-Weights.txt')
        attention.v_proj_weight = load_parameter(f'{prefix}_value_matrix-Weights.txt')
        attention.v_proj_bias = load_parameter(f'{prefix}_value_bias-Weights.txt')
        attention.out_proj_weight = load_parameter(f'{prefix}_output_matrix-Weights.txt')
        attention.out_proj_bias = load_parameter(f'{prefix}_output_bias-Weights.txt')

        # Load weights for feedforward network
        prefix = f'{weights_prefix}-transformer_encoder{i}'
        layer.linear1.weight = load_parameter(f'{prefix}_fc1_matrix-Weights.txt')
        layer.linear1.bias = load_parameter(f'{prefix}_fc1_bias-Weights.txt')
        layer.linear2.weight = load_parameter(f'{prefix}_fc2_matrix-Weights.txt')
        layer.linear2.bias = load_parameter(f'{prefix}_fc2_bias-Weights.txt')

    # Load weights for decoder
    for i, layer in enumerate(transformer.decoder.layers):

        # Load weights for self-attention
        attention = layer.self_attn
        attention._qkv_same_embed_dim = False
        prefix = f'{weights_prefix}-transformer_decoder{i}_attention1'
        attention.q_proj_weight = load_parameter(f'{prefix}_query_matrix-Weights.txt')
        attention.q_proj_bias = load_parameter(f'{prefix}_query_bias-Weights.txt')
        attention.k_proj_weight = load_parameter(f'{prefix}_key_matrix-Weights.txt')
        attention.k_proj_bias = load_parameter(f'{prefix}_key_bias-Weights.txt')
        attention.v_proj_weight = load_parameter(f'{prefix}_value_matrix-Weights.txt')
        attention.v_proj_bias = load_parameter(f'{prefix}_value_bias-Weights.txt')
        attention.out_proj_weight = load_parameter(f'{prefix}_output_matrix-Weights.txt')
        attention.out_proj_bias = load_parameter(f'{prefix}_output_bias-Weights.txt')

        # Load weights for attention with memory
        attention = layer.multihead_attn
        attention._qkv_same_embed_dim = False
        prefix = f'{weights_prefix}-transformer_decoder{i}_attention2'
        attention.q_proj_weight = load_parameter(f'{prefix}_query_matrix-Weights.txt')
        attention.q_proj_bias = load_parameter(f'{prefix}_query_bias-Weights.txt')
        attention.k_proj_weight = load_parameter(f'{prefix}_key_matrix-Weights.txt')
        attention.k_proj_bias = load_parameter(f'{prefix}_key_bias-Weights.txt')
        attention.v_proj_weight = load_parameter(f'{prefix}_value_matrix-Weights.txt')
        attention.v_proj_bias = load_parameter(f'{prefix}_value_bias-Weights.txt')
        attention.out_proj_weight = load_parameter(f'{prefix}_output_matrix-Weights.txt')
        attention.out_proj_bias = load_parameter(f'{prefix}_output_bias-Weights.txt')

        # Load weights for feedforward network
        prefix = f'{weights_prefix}-transformer_decoder{i}'
        layer.linear1.weight = load_parameter(f'{prefix}_fc1_matrix-Weights.txt')
        layer.linear1.bias = load_parameter(f'{prefix}_fc1_bias-Weights.txt')
        layer.linear2.weight = load_parameter(f'{prefix}_fc2_matrix-Weights.txt')
        layer.linear2.bias = load_parameter(f'{prefix}_fc2_bias-Weights.txt')

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

def evaluate_transformer(weights_prefix):
    """Evaluate transformer model with weights from LBANN."""

    # Load model weights from file
    # TODO: Use embeddings for classifier
    embedding_layer = load_embedding_layer(weights_prefix)
    transformer = load_transformer(weights_prefix)
    classifier = torch.nn.Linear(embed_dim, vocab_size)
    classifier.weight = load_parameter(
        f'{weights_prefix}-classifier_matrix-Weights.txt'
    )
    classifier.bias = load_parameter(
        f'{weights_prefix}-classifier_bias-Weights.txt'
    )

    # Evaluate model
    # TODO: Greedy decoding
    bleu_scores = []
    for batch, index_start in enumerate(range(0, num_samples, mini_batch_size)):
        index_end = min(index_start+mini_batch_size, num_samples)

        # Get embeddings
        indices = list(range(index_start, index_end))
        tokens_en, tokens_de = get_batch(indices)
        embeddings_en = embedding_layer(tokens_en)
        embeddings_de = embedding_layer(tokens_de[:-1,:])

        # Apply transformer
        preds = transformer(
            add_positional_encoding(embeddings_en),
            add_positional_encoding(embeddings_de),
            tgt_mask=transformer.generate_square_subsequent_mask(embeddings_de.shape[0]),
        )

        # Predict outputs
        preds = classifier(preds)
        preds = preds.argmax(dim=2)

        # Compute BLEU score
        for i in range(preds.shape[1]):
            hypothesis = dataset.detokenize(preds[:,i].numpy())
            reference = dataset.detokenize(tokens_de[:,i].numpy())
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

if __name__ == "__main__":

    # Command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'weights_prefix', type=str,
        help='prefix for saved weights from LBANN')
    args = parser.parse_args()

    # Evaluate model
    evaluate_transformer(args.weights_prefix)

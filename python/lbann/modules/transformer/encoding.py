"""Input encoding modules for transformer models."""
import math
import numpy as np
from typing import Tuple
import lbann
from ..base import Module


class SequenceEncoding:
    """
    Abstract class that implements attention-based sequence encoding schemes.
    There are two methods that can be (optionally) implemented: ``apply_input``,
    which applies on transformer inputs immediately after embedding; and
    ``apply_layer``, which is applied in every encoder/decoder and is given
    the layer type and index.
    """

    def apply_input(self, x: lbann.Layer, length: int,
                    **extra_kwargs) -> lbann.Layer:
        """
        Applies sequence encoding on the input of a transformer, immediately
        after token embedding.

        :param x: The output of the embedded sequence minibatch.
        :param length: Sequence length.
        :param extra_kwargs: Additional arguments to pass to each internal Layer.
        :return: Encoded input.
        """
        return x  # Do nothing

    def apply_layer(
            self, q: lbann.Layer, k: lbann.Layer, v: lbann.Layer, length: int,
            **extra_kwargs) -> Tuple[lbann.Layer, lbann.Layer, lbann.Layer]:
        """
        Applies sequence encoding within a transformer encoder/decoder layer.
        Encoding is performed on each transformer layer's multi-head attention
        inputs, after obtaining their hidden representation (e.g., ``WQ @ q``).

        :param q: The input queries of the transformer layer.
        :param k: The input keys of the transformer layer.
        :param v: The input values of the transformer layer.
        :param length: Sequence length.
        :param extra_kwargs: Additional arguments to pass to each internal Layer.
        :return: Encoded tuple of (q, k, v).
        """
        return q, k, v  # Do nothing


def _make_constant_from_array(array, name=None) -> lbann.WeightsLayer:
    """
    Helper function that creates a constant tensor in LBANN from a given numpy
    array.
    """
    if name is not None:
        weights_name = name + '_weights'
    else:
        weights_name = None

    w = lbann.Weights(
        initializer=lbann.ValueInitializer(values=array.flat),
        optimizer=lbann.NoOptimizer(),
        name=weights_name,
    )
    return lbann.WeightsLayer(
        dims=array.shape,
        weights=w,
        name=name,
    )


class PositionalEncoding(SequenceEncoding):
    """
    Implements positional encoding, as defined by Vaswani et al.,
    "Attention Is All You Need" (2017).
    """
    global_count = 0  # Static instance counter

    def __init__(
        self,
        embed_dim,
        dropout=0.0,
        name=None,
    ):
        # Module name
        PositionalEncoding.global_count += 1
        self.instance = 0
        self.name = name
        if not self.name:
            self.name = f'posenc{PositionalEncoding.global_count}'

        # Parameters
        self._positional_encoding_cache = {}
        self.embed_dim = embed_dim
        self.dropout_prob = dropout

    def _positional_encoding(self, sequence_length):
        """Positional encodings corresponding to a sequence length.

        PE(pos,2*i)   = sin( pos / 10000**(2*i/embed_dim) )

        PE(pos,2*i+1) = cos( pos / 10000**(2*i/embed_dim) )

        Encodings are memoized.

        """

        # Construct positional encoding if not in cache
        if sequence_length not in self._positional_encoding_cache:
            vals = []
            for pos in range(sequence_length):
                for i in range((self.embed_dim + 1) // 2):
                    x = pos / 10000**(2 * i / self.embed_dim)
                    vals.append(math.sin(x))
                    vals.append(math.cos(x))
                if self.embed_dim % 2 != 0:
                    vals.pop()

            self._positional_encoding_cache[
                sequence_length] = _make_constant_from_array(
                    np.array(vals).reshape([sequence_length, self.embed_dim]),
                    name=f'{self.name}_positional{sequence_length}',
                )

        # Return cached positional encoding
        return self._positional_encoding_cache[sequence_length]

    def apply_input(self, inputs, input_length, **extra_kwargs):
        self.instance += 1

        result = lbann.Add(
            inputs,
            self._positional_encoding(input_length),
            name=f'{self.name}_instance{self.instance}_peadd',
            **extra_kwargs,
        )

        # Input dropout
        if self.dropout_prob > 0:
            return lbann.Dropout(
                result,
                keep_prob=1 - self.dropout_prob,
                name=f'{self.name}_pedrop',
                **extra_kwargs,
            )
        return result


class LearnedInputEncoding(SequenceEncoding):
    """
    Implements learned input encoding (via embeddings), as used in GPT-style
    transformers.
    """
    global_count = 0  # Static instance counter

    def __init__(
        self,
        embed_dim,
        max_sequence_length,
        dropout=0.0,
        name=None,
    ):
        # Module name
        LearnedInputEncoding.global_count += 1
        self.instance = 0
        self.name = name
        if not self.name:
            self.name = f'learnedenc{LearnedInputEncoding.global_count}'

        # Parameters
        self._positional_encoding_cache = {}
        self.embed_dim = embed_dim
        self.dropout_prob = dropout
        self.max_sequence_length = max_sequence_length

        self.encoding_weights = lbann.Weights(
            name=self.name + '_weights',
            initializer=lbann.NormalInitializer(standard_deviation=0.01),
        )
        self.position_ids = _make_constant_from_array(
            np.arange(max_sequence_length))

    def compute_embeddings(self):
        return lbann.Embedding(
            self.position_ids,
            weights=self.encoding_weights,
            num_embeddings=self.max_sequence_length,
            embedding_dim=self.embed_dim,
        )

    def apply_input(self,
                    inputs,
                    input_length,
                    learned_encoding=None,
                    **extra_kwargs):
        self.instance += 1

        if learned_encoding is None:
            learned_encoding = self.compute_embeddings()

        # Subsegment learned encodings if shorter than sequence length
        if input_length < self.max_sequence_length:
            learned_encoding = lbann.Identity(
                lbann.Slice(learned_encoding,
                            axis=0,
                            slice_points=[0, input_length],
                            **extra_kwargs), **extra_kwargs)

        result = lbann.Add(
            inputs,
            learned_encoding,
            name=f'{self.name}_instance{self.instance}_peadd',
            **extra_kwargs,
        )

        # Input dropout
        if self.dropout_prob > 0:
            return lbann.Dropout(
                result,
                keep_prob=1 - self.dropout_prob,
                name=f'{self.name}_pedrop',
                **extra_kwargs,
            )
        return result


class RotaryPositionalEmbedding(SequenceEncoding):
    """
    Implements Rotary Positional Embedding (RoPE).

    Based on Jianlin Su et al. "RoFormer: Enhanced Transformer with Rotary
    Position Embedding" (2021). arXiv:2104.09864.
    """
    global_count = 0  # Static instance counter

    def __init__(
        self,
        freq_dim: int,
        max_sequence_length: int,
        num_heads: int,
        embed_values: bool = False,
        theta: float = 10000.0,
        name: str = None,
    ):
        """
        Initializes rotary positional embeddings.

        :param freq_dim: The dimensionality of the frequency vector. Must be
                         even. By default, it is the dimensionality of the
                         queries/keys (``embed_dim // num_heads``).
        :param max_sequence_length: Largest sequence length used in training.
                                    Used for caching precomputed values.
        :param num_heads: Number of heads in the transformer.
        :param embed_values: If True, embeds the values as well as the queries
                             and keys.
        :param theta: The base of the frequencies (default: 10000; from paper)
        :param name: Optional name specification.
        """
        # Module name
        RotaryPositionalEmbedding.global_count += 1
        self.instance = 0
        self.name = name
        if not self.name:
            self.name = f'rope{RotaryPositionalEmbedding.global_count}'
        if freq_dim % 2 == 1:
            raise ValueError('With rotary positional embedding, the number of '
                             f'frequencies must be even. Got {freq_dim}')

        # Parameters
        self.embed_values = embed_values
        self.dim = freq_dim
        self.theta = theta
        self.max_sequence_length = max_sequence_length
        self.num_heads = num_heads

        # Precompute tensors
        freq = np.arange(0, self.dim, 2) / self.dim
        self.inv_freq = 1 / (theta**freq)
        self.cos, self.sin = self._precompute_frequencies(max_sequence_length)

    def _precompute_frequencies(self, sequence_length: int):
        t = np.arange(sequence_length)
        freq = np.outer(t, self.inv_freq)
        emb = np.concatenate([freq, freq], axis=-1)
        # Add new axis for "broadcasting" (output shape should be SxE for
        # S=sequence length and E=embedding dimension)
        cos = np.tile(np.cos(emb), self.num_heads)
        sin = np.tile(np.sin(emb), self.num_heads)
        return (
            _make_constant_from_array(cos, f'rope_cos_{sequence_length}'),
            _make_constant_from_array(sin, f'rope_sin_{sequence_length}'),
        )

    def _rotate_half(self, x: lbann.Layer, length: int, **extra_kwargs):
        """
        Helper method that rotates half of a tensor x.
        """
        # SxE -> SxHxP
        r = lbann.Reshape(x,
                          dims=(length, self.num_heads, self.dim),
                          **extra_kwargs)
        s = lbann.Slice(r,
                        slice_points=[0, self.dim // 2, self.dim],
                        axis=2,
                        **extra_kwargs)
        x1 = lbann.Identity(s, **extra_kwargs)
        x2 = lbann.Identity(s, **extra_kwargs)
        nx2 = lbann.Scale(x2, constant=-1, **extra_kwargs)
        cat = lbann.Concatenation([nx2, x1], axis=2, **extra_kwargs)

        # Reshape back to SxE
        return lbann.Reshape(cat,
                             dims=(length, self.num_heads * self.dim),
                             **extra_kwargs)

    def _embed(self, x: lbann.Layer, length: int, sliced_cos: lbann.Layer,
               sliced_sin: lbann.Layer, **extra_kwargs):
        """
        Helper method that applies rotary embeddings on a tensor x.
        """
        rot = self._rotate_half(x, length, **extra_kwargs)
        return lbann.Add(
            lbann.Multiply(x, sliced_cos, **extra_kwargs),
            lbann.Multiply(rot, sliced_sin, **extra_kwargs),
            **extra_kwargs,
        )

    def apply_layer(
            self, q: lbann.Layer, k: lbann.Layer, v: lbann.Layer, length: int,
            **extra_kwargs) -> Tuple[lbann.Layer, lbann.Layer, lbann.Layer]:
        # If length is not given, maximum sequence length is assumed
        if length is None:
            length = self.max_sequence_length

        if length == self.max_sequence_length:
            sliced_cos = self.cos
            sliced_sin = self.sin
        else:
            sliced_cos = lbann.Identity(
                lbann.Slice(self.cos,
                            slice_points=[0, length],
                            axis=0,
                            **extra_kwargs), **extra_kwargs)
            sliced_sin = lbann.Identity(
                lbann.Slice(self.sin,
                            slice_points=[0, length],
                            axis=0,
                            **extra_kwargs), **extra_kwargs)

        eq = self._embed(q, length, sliced_cos, sliced_sin, **extra_kwargs)
        ek = self._embed(k, length, sliced_cos, sliced_sin, **extra_kwargs)

        if self.embed_values:
            ev = self._embed(v, length, sliced_cos, sliced_sin, **extra_kwargs)
        else:
            ev = v

        return eq, ek, ev

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

    def apply_input(self, x: lbann.Layer, length: int) -> lbann.Layer:
        """
        Applies sequence encoding on the input of a transformer, immediately
        after token embedding.

        :param x: The output of the embedded sequence minibatch.
        :param length: Sequence length.
        :return: Encoded input.
        """
        return x  # Do nothing

    def apply_layer(
            self, q: lbann.Layer, k: lbann.Layer,
            v: lbann.Layer) -> Tuple[lbann.Layer, lbann.Layer, lbann.Layer]:
        """
        Applies sequence encoding within a transformer encoder/decoder layer.
        Encoding is performed on each transformer layer's multi-head attention
        inputs, after obtaining their hidden representation (e.g., ``WQ @ q``).

        :param q: The input queries of the transformer layer.
        :param k: The input keys of the transformer layer.
        :param v: The input values of the transformer layer.
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

    def apply_input(self, inputs, input_length):
        self.instance += 1

        result = lbann.Add(
            inputs,
            self._positional_encoding(input_length),
            name=f'{self.name}_instance{self.instance}_peadd',
        )

        # Input dropout
        if self.dropout_prob > 0:
            return lbann.Dropout(
                result,
                keep_prob=1 - self.dropout_prob,
                name=f'{self.name}_pedrop',
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

    def apply_input(self, inputs, input_length, learned_encoding=None):
        self.instance += 1

        if learned_encoding is None:
            learned_encoding = self.compute_embeddings()

        # Subsegment learned encodings if shorter than sequence length
        if input_length < self.max_sequence_length:
            learned_encoding = lbann.Identity(
                lbann.Slice(learned_encoding,
                            axis=0,
                            slice_points=[0, input_length]))

        result = lbann.Add(
            inputs,
            learned_encoding,
            name=f'{self.name}_instance{self.instance}_peadd',
        )

        # Input dropout
        if self.dropout_prob > 0:
            return lbann.Dropout(
                result,
                keep_prob=1 - self.dropout_prob,
                name=f'{self.name}_pedrop',
            )
        return result

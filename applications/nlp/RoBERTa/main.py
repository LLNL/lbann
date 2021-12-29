import numpy as np
import math
import sys
import os

import lbann
import lbann.modules
from lbann.util import str_list

from lbann_models.loss import CrossEntropyLoss

_Permute_cache = {}
_Cumsum_cache = {}


class RobertaEmbeddings(lbann.modules.Module):
    def __init__(self, config, name, load_weights=True):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.pad_token_id = config.pad_token_id
        self.padding_idx = config.pad_token_id
        self.max_position_embeddings = config.max_position_embeddings
        self.type_vocab_size = config.type_vocab_size
        self.layer_norm_eps = config.layer_norm_eps
        self.hidden_dropout_prob = config.hidden_dropout_prob
        self.position_embedding_type = getattr(
            config, "position_embedding_type", "absolute"
        )
        self.input_shape = config.input_shape
        self.name = name
        self.load_weights = load_weights

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
    ):

        if position_ids is None:
            if input_ids is not None:
                position_ids = create_position_ids_from_input_ids(
                    input_ids,
                    self.input_shape,
                    self.padding_idx,
                )
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(
                    inputs_embeds
                )

        if token_type_ids is None:
            token_type_ids = lbann.Constant(
                value=0, num_neurons=str_list(self.input_shape)
            )

        if inputs_embeds is None:
            inputs_embeds = lbann.Embedding(
                input_ids,
                num_embeddings=self.vocab_size,
                embedding_dim=self.hidden_size,
                padding_idx=self.pad_token_id,
                weights=_load_pretrained_weights(
                    ".".join((self.name, "word_embeddings.weight")),
                    load_weights=self.load_weights,
                ),
                name=".".join((self.name, "word_embeddings")),
            )
        token_type_embeddings = lbann.Embedding(
            token_type_ids,
            num_embeddings=self.type_vocab_size,
            embedding_dim=self.hidden_size,
            weights=_load_pretrained_weights(
                ".".join((self.name, "token_type_embeddings.weight")),
                load_weights=self.load_weights,
            ),
            name=".".join((self.name, "token_type_embeddings")),
        )

        embeddings = lbann.Add(inputs_embeds, token_type_embeddings)
        if self.position_embedding_type == "absolute":
            position_embeddings = lbann.Embedding(
                position_ids,
                num_embeddings=self.max_position_embeddings,
                embedding_dim=self.hidden_size,
                padding_idx=self.pad_token_id,
                weights=_load_pretrained_weights(
                    ".".join((self.name, "position_embeddings.weight")),
                    load_weights=self.load_weights,
                ),
                name=".".join((self.name, "position_embeddings")),
            )
            embeddings = lbann.Add(embeddings, position_embeddings)

        embeddings = _LayerNorm(
            embeddings,
            self.layer_norm_eps,
            self.input_shape + (self.hidden_size,),
            weights=_load_pretrained_weights(
                ".".join((self.name, "layernorm.weightbias")),
                load_weights=self.load_weights,
            ),
            name=".".join((self.name, "LayerNorm")),
        )
        embeddings = lbann.Dropout(embeddings, keep_prob=self.hidden_dropout_prob)
        return embeddings


class RobertaEncoder(lbann.modules.Module):
    def __init__(self, config, name, load_weights=True):
        super().__init__()
        self.config = config
        self.input_shape = config.input_shape + (config.hidden_size,)
        self.name = name
        self.load_weights = load_weights

        self.layer = [
            RobertaLayer(
                config, ".".join((name, "layer", str(i))), load_weights=load_weights
            )
            for i in range(config.num_hidden_layers)
        ]

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
    ):
        for i, layer_module in enumerate(self.layer):
            layer_head_mask = head_mask[i] if head_mask is not None else None

            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                layer_head_mask,
            )

            hidden_states = layer_outputs

        return (hidden_states,)


class RobertaLayer(lbann.modules.Module):
    def __init__(self, config, name, load_weights=True):
        super().__init__()
        self.input_shape = config.input_shape + (config.hidden_size,)
        self.name = name
        self.load_weights = load_weights

        self.attention = RobertaAttention(
            config, ".".join((name, "attention")), load_weights=load_weights
        )
        self.intermediate = RobertaIntermediate(
            config, ".".join((name, "intermediate")), load_weights=load_weights
        )
        self.output = RobertaOutput(
            config, ".".join((name, "output")), load_weights=load_weights
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
    ):
        hidden_states = lbann.Reshape(hidden_states, dims=str_list(self.input_shape))
        attention_output = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
        )
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)

        return layer_output


class RobertaSelfAttention(lbann.modules.Module):
    def __init__(self, config, name, load_weights=True):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(
            config, "embedding_size"
        ):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        self.batch_size = config.batch_size
        self.input_shape = config.input_shape + (config.hidden_size,)
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.attention_probs_dropout_prob = config.attention_probs_dropout_prob
        self.position_embedding_type = getattr(
            config, "position_embedding_type", "absolute"
        )
        self.name = name
        self.load_weights = load_weights

    def transpose_for_scores(self, x, dims):
        new_x_shape = dims[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = lbann.Reshape(x, dims=lbann.util.str_list(new_x_shape))
        return _Permute(x, new_x_shape, axes=(0, 2, 1, 3))

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
    ):
        mixed_query_layer, query_shape = _Linear(
            hidden_states,
            self.input_shape,
            self.all_head_size,
            weights=_load_pretrained_weights(
                ".".join((self.name, "query.weight")),
                ".".join((self.name, "query.bias")),
                load_weights=self.load_weights,
            ),
            name=".".join((self.name, "query")),
        )
        query_layer, query_shape = self.transpose_for_scores(
            mixed_query_layer, query_shape
        )

        key_layer, key_shape = _Linear(
            hidden_states,
            self.input_shape,
            self.all_head_size,
            weights=_load_pretrained_weights(
                ".".join((self.name, "key.weight")),
                ".".join((self.name, "key.bias")),
                load_weights=self.load_weights,
            ),
            name=".".join((self.name, "key")),
        )
        key_layer, key_shape = self.transpose_for_scores(key_layer, key_shape)
        value_layer, value_shape = _Linear(
            hidden_states,
            self.input_shape,
            self.all_head_size,
            weights=_load_pretrained_weights(
                ".".join((self.name, "value.weight")),
                ".".join((self.name, "value.bias")),
                load_weights=self.load_weights,
            ),
            name=".".join((self.name, "value")),
        )
        value_layer, value_shape = self.transpose_for_scores(value_layer, value_shape)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        key_layer, key_shape = _Permute(key_layer, key_shape, axes=(0, 1, -1, -2))
        attention_scores, attention_shape = _Matmul(
            query_layer, query_shape, key_layer, key_shape
        )

        attention_scores = lbann.Scale(
            attention_scores, constant=1 / math.sqrt(self.attention_head_size)
        )

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in RobertaModel forward() function)
            attention_scores = lbann.Add(attention_scores, attention_mask)

        # Normalize the attention scores to probabilities.
        attention_scores = lbann.Reshape(
            attention_scores,
            dims=str_list([np.prod(attention_shape[:-1]), attention_shape[-1]]),
        )
        attention_probs = lbann.ChannelwiseSoftmax(attention_scores)
        attention_probs = lbann.Reshape(attention_probs, dims=str_list(attention_shape))

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = lbann.Dropout(
            attention_probs,
            keep_prob=self.attention_probs_dropout_prob,
        )

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = lbann.Multiply(attention_probs, head_mask)

        context_layer, context_shape = _Matmul(
            attention_probs, attention_shape, value_layer, value_shape
        )
        context_layer, context_shape = _Permute(
            context_layer, context_shape, axes=(0, 2, 1, 3)
        )
        new_context_layer_shape = context_shape[:-2] + (self.all_head_size,)
        context_layer = lbann.Reshape(context_layer, dims=str_list(self.input_shape))

        return context_layer


class RobertaSelfOutput(lbann.modules.Module):
    def __init__(self, config, name, load_weights=True):
        super().__init__()
        self.input_shape = config.input_shape + (config.hidden_size,)
        self.hidden_size = config.hidden_size
        self.layer_norm_eps = config.layer_norm_eps
        self.hidden_dropout_prob = config.hidden_dropout_prob
        self.name = name
        self.load_weights = load_weights

    def forward(self, hidden_states, input_tensor):
        hidden_states, hidden_shape = _Linear(
            hidden_states,
            self.input_shape,
            self.hidden_size,
            weights=_load_pretrained_weights(
                ".".join((self.name, "dense.weight")),
                ".".join((self.name, "dense.bias")),
                load_weights=self.load_weights,
            ),
            name=".".join((self.name, "dense")),
        )
        hidden_states = lbann.Dropout(hidden_states, keep_prob=self.hidden_dropout_prob)
        hidden_states = _LayerNorm(
            lbann.Add(hidden_states, input_tensor),
            self.layer_norm_eps,
            hidden_shape,
            weights=_load_pretrained_weights(
                ".".join((self.name, "layernorm.weightbias")),
                load_weights=self.load_weights,
            ),
            name=".".join((self.name, "LayerNorm")),
        )
        return hidden_states


class RobertaAttention(lbann.modules.Module):
    def __init__(self, config, name, load_weights=True):
        super().__init__()
        self.input_shape = config.input_shape + (config.hidden_size,)
        self.name = name
        self.load_weights = load_weights

        self.self = RobertaSelfAttention(
            config, ".".join((name, "self")), load_weights=load_weights
        )
        self.output = RobertaSelfOutput(
            config, ".".join((name, "output")), load_weights=load_weights
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
    ):
        self_output = self.self(
            hidden_states,
            attention_mask,
            head_mask,
        )
        attention_output = self.output(self_output, hidden_states)
        return attention_output


class RobertaIntermediate(lbann.modules.Module):
    def __init__(self, config, name, load_weights=True):
        super().__init__()
        self.intermediate_size = config.intermediate_size
        self.input_shape = config.input_shape + (config.hidden_size,)
        self.name = name
        self.load_weights = load_weights
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states, hidden_shape = _Linear(
            hidden_states,
            self.input_shape,
            self.intermediate_size,
            weights=_load_pretrained_weights(
                ".".join((self.name, "dense.weight")),
                ".".join((self.name, "dense.bias")),
                load_weights=self.load_weights,
            ),
            name=".".join((self.name, "dense")),
        )
        hidden_states = self.intermediate_act_fn(hidden_states, hidden_shape)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertOutput
class RobertaOutput(lbann.modules.Module):
    def __init__(self, config, name, load_weights=True):
        super().__init__()
        self.input_shape = config.input_shape + (config.intermediate_size,)
        self.hidden_size = config.hidden_size
        self.hidden_dropout_prob = config.hidden_dropout_prob
        self.layer_norm_eps = config.layer_norm_eps
        self.name = name
        self.load_weights = load_weights

    def forward(self, hidden_states, input_tensor):
        hidden_states, hidden_shape = _Linear(
            hidden_states,
            self.input_shape,
            self.hidden_size,
            weights=_load_pretrained_weights(
                ".".join((self.name, "dense.weight")),
                ".".join((self.name, "dense.bias")),
                load_weights=self.load_weights,
            ),
            name=".".join((self.name, "dense")),
        )
        hidden_states = lbann.Dropout(hidden_states, keep_prob=self.hidden_dropout_prob)
        hidden_states = _LayerNorm(
            lbann.Add(hidden_states, input_tensor),
            self.layer_norm_eps,
            hidden_shape,
            weights=_load_pretrained_weights(
                ".".join((self.name, "layernorm.weightbias")),
                load_weights=self.load_weights,
            ),
            name=".".join((self.name, "LayerNorm")),
        )
        return hidden_states


class RobertaPooler(lbann.modules.Module):
    def __init__(self, config, name, load_weights=True):
        super().__init__()
        self.input_shape = config.input_shape + (config.hidden_size,)
        self.hidden_size = config.hidden_size
        self.name = name
        self.load_weights = load_weights

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = lbann.Slice(
            hidden_states, axis=1, slice_points=str_list([0, 1])
        )
        pooled_output, pooled_output_shape = _Linear(
            first_token_tensor,
            (self.input_shape[0], self.input_shape[-1]),
            self.hidden_size,
            weights=_load_pretrained_weights(
                ".".join((self.name, "dense.weight")),
                ".".join((self.name, "dense.bias")),
                load_weights=self.load_weights,
            ),
            name=".".join((self.name, "dense")),
        )
        pooled_output = lbann.Tanh(
            pooled_output, name=".".join((self.name, "activation"))
        )
        return pooled_output


class RobertaModel(lbann.modules.Module):
    def __init__(self, config, add_pooling_layer=True, load_weights=True):
        self.config = config

        self.embeddings = RobertaEmbeddings(
            config, "embeddings", load_weights=load_weights
        )
        self.encoder = RobertaEncoder(config, "encoder", load_weights=load_weights)

        self.pooler = (
            RobertaPooler(config, "pooler", load_weights=load_weights)
            if add_pooling_layer
            else None
        )
        self.input_shape = config.input_shape
        self.attn_mask_shape = (
            config.input_shape[0],
            config.num_attention_heads,
            config.input_shape[1],
            config.input_shape[1],
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
    ):

        if attention_mask is None:
            attention_mask = lbann.Constant(
                value=1, num_neurons=str_list(self.attn_mask_shape)
            )

        if token_type_ids is None:
            token_type_ids = lbann.Constant(
                value=0, num_neurons=str_list(self.input_shape)
            )

        if head_mask is None:
            head_mask = [None] * self.config.num_hidden_layers

        input_ids = lbann.Reshape(input_ids, dims=str_list(self.input_shape))
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )
        embedding_output = lbann.Reshape(
            embedding_output, dims=str_list(self.input_shape + (768,))
        )
        encoder_output = self.encoder(
            embedding_output,
            attention_mask=attention_mask,
            head_mask=head_mask,
        )
        pooled_output = self.pooler(encoder_output) if self.pooler is not None else None

        return pooled_output


def _relu(x, _):
    return lbann.Relu(x)


def _silu(x, _):
    return lbann.Multiply(x, lbann.Sigmoid(x))


def _gelu(x, x_shape):
    # return 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x ** 3)))
    # Based on: https://github.com/huggingface/transformers/blob/38a716cd41f22f6a7d5ff3dc081903090198803a/examples/research_projects/bertabs/modeling_bertabs.py#L658
    sqrt_2_over_pi = math.sqrt(2 / math.pi)
    b_coef = 0.044715
    x_cubed = lbann.Multiply(lbann.Multiply(lbann.Identity(x), x), x)
    inner_tanh_x_comp = lbann.Add(x, lbann.Scale(x_cubed, constant=b_coef))
    tanh_x = lbann.Tanh(lbann.Scale(inner_tanh_x_comp, constant=sqrt_2_over_pi))
    return lbann.Scale(
        lbann.Multiply(x, lbann.AddConstant(tanh_x, constant=1)), constant=0.5
    )


def _tanh(x, _):
    return lbann.Tanh(x)


def _sigmoid(x, _):
    return lbann.Sigmoid(x)


ACT2FN = {
    "relu": _relu,
    "silu": _silu,
    "swish": _silu,
    "gelu": _gelu,
    "tanh": _tanh,
    "sigmoid": _sigmoid,
}


def create_position_ids_from_input_ids(
    input_ids, input_shape, padding_idx, past_key_values_length=0
):
    padding_idx = lbann.Constant(value=padding_idx, num_neurons=str_list(input_shape))
    mask = lbann.NotEqual(input_ids, padding_idx)
    incremental_indices = _Cumsum(mask, input_shape, axis=1)
    past_key_values_length = lbann.Constant(
        value=past_key_values_length, num_neurons=str_list(input_shape)
    )
    incremental_indices = lbann.Add(incremental_indices, past_key_values_length)
    incremental_indices = lbann.Multiply(incremental_indices, mask)
    incremental_indices = lbann.Add(incremental_indices, padding_idx)

    return incremental_indices


def _Permute(x, dims, axes=None, name=""):
    global _Permute_cache
    key = (dims, axes)
    size = np.prod(dims)
    if key not in _Permute_cache:
        # Construct gather indices
        inds = np.arange(size).reshape(dims, order="C").transpose(axes)
        inds = lbann.Weights(
            initializer=lbann.ValueInitializer(
                values=str_list(np.nditer(inds, order="C")),
            ),
            optimizer=lbann.NoOptimizer(),
        )
        inds = lbann.WeightsLayer(dims=str_list([size]), weights=inds)
        _Permute_cache[key] = inds

    # Apply transpose with gather
    inds = _Permute_cache[key]
    if axes == None:
        new_dims = dims[::-1]
    else:
        new_dims = np.array(dims)[list(axes)]
    x = lbann.Reshape(x, dims=str_list([size]))
    y = lbann.Gather(x, inds)
    y = lbann.Reshape(y, dims=str_list(list(new_dims)), name=name)

    return y, tuple(new_dims)


def _Cumsum(x, dims, axis=0):
    global _Cumsum_cache

    if len(dims) != 2:
        raise RuntimeError("dims > 2 not tested/supported for cumsum")
    if (axis < 0) or (axis > 1):
        raise RuntimeError("Unsupported cumsum axis: {}".format(axis))
    shape = (dims[axis], dims[axis])
    if shape not in _Cumsum_cache:
        tril_ones = np.tril(np.full(shape, 1, dtype=int), k=0)
        tril_ones = lbann.Weights(
            initializer=lbann.ValueInitializer(
                values=str_list(np.nditer(tril_ones, order="C")),
            ),
            optimizer=lbann.NoOptimizer(),
        )
        tril_ones = lbann.WeightsLayer(dims=str_list(shape), weights=tril_ones)
        _Cumsum_cache[shape] = tril_ones

    # Apply cumsum
    tril_ones = _Cumsum_cache[shape]
    if axis == 0:
        x = lbann.MatMul(tril_ones, x)
        return x
    if axis == 1:
        x = lbann.MatMul(x, tril_ones, transpose_b=True)
        return x


def _load_pretrained_weights_layer(
    fn,
    file_dir=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "pretrained_weights"
    ),
):
    weights_file = os.path.join(file_dir, fn + ".npy")
    dims = np.load(weights_file).shape
    weights = _load_pretrained_weights(fn, file_dir)
    weights = lbann.WeightsLayer(weights=weights, dims=str_list(dims))
    return weights, dims


def _load_pretrained_weights(
    *fn,
    file_dir=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "pretrained_weights"
    ),
    load_weights=True,
):
    if not load_weights:
        return []

    weights = []
    for f in fn:
        w_file = os.path.join(file_dir, f + ".npy")
        weights.append(lbann.Weights(initializer=lbann.NumpyInitializer(file=w_file)))

    if len(weights) == 1:
        weights = weights[0]
    return weights


# Mimics torch.matmul in LBANN
def _Matmul(x, x_shape, y, y_shape):
    if len(x_shape) != len(y_shape):
        raise RuntimeError(
            "Broadcasting not fully implemented, tensors must have same dimension"
        )
    need_reshape = (len(x_shape) > 3) and (len(y_shape) > 3)
    if need_reshape:
        if x_shape[:-2] != y_shape[:-2]:
            raise RuntimeError("The first n-2 dimensions must match")
        new_x_shape = (np.prod(x_shape[:-2]),) + x_shape[-2:]
        x = lbann.Reshape(x, dims=str_list(new_x_shape))

        new_y_shape = (np.prod(y_shape[:-2]),) + y_shape[-2:]
        y = lbann.Reshape(y, dims=str_list(new_y_shape))

    z = lbann.MatMul(x, y)

    z_shape = x_shape[:-1] + (y_shape[-1],)
    if need_reshape:
        z = lbann.Reshape(z, dims=str_list(z_shape))

    return z, z_shape


# Mimics torch.nn.Linear in LBANN
def _Linear(x, input_shape, hidden_size, weights=[], name=""):
    need_reshape = len(input_shape) > 2
    if need_reshape:
        new_in_shape = (np.prod(input_shape[:-1]), input_shape[-1])
        x = lbann.Reshape(x, dims=str_list(new_in_shape))

    y = lbann.ChannelwiseFullyConnected(
        x, output_channel_dims=[hidden_size], weights=weights, name=name
    )

    if need_reshape:
        new_out_shape = input_shape[:-1] + (hidden_size,)
        y = lbann.Reshape(y, dims=str_list(new_out_shape))
    else:
        new_out_shape = (input_shape[0], hidden_size)

    return y, new_out_shape


# Mimics torch.nn.layernorm in LBANN
def _LayerNorm(x, epsilon, input_shape, weights=[], name=""):
    if len(input_shape) > 2:
        x = lbann.Reshape(
            x, dims=str_list([np.prod(input_shape[:-1]), input_shape[-1]])
        )
    x = lbann.InstanceNorm(x, epsilon=epsilon)
    x = lbann.Reshape(x, dims=str_list(input_shape))
    if weights is not []:
        x, new_x_shape = _Permute(x, input_shape)
        x = lbann.ChannelwiseScaleBias(x, weights=weights)
        x, _ = _Permute(x, new_x_shape, name=name)

    return x


def make_model(config):
    # Input data
    input_ = lbann.Slice(
        lbann.Input(data_field="samples"), slice_points=str_list([0, 1, 1 + 16 * 32])
    )
    labels = lbann.Identity(input_)
    sample = lbann.Reshape(input_, dims=str_list([16, 32]))

    # Model
    model = RobertaModel(config, load_weights=True)
    out = model(sample)
    out = lbann.ChannelwiseFullyConnected(out, output_channel_dims=[1000])

    loss = CrossEntropyLoss(10, data_layout="model_parallel")
    obj = loss(out, labels)
    metrics = [lbann.Metric(obj, name="loss")]

    return lbann.Model(
        0,
        layers=lbann.traverse_layer_graph(input_),
        objective_function=obj,
        metrics=metrics,
        callbacks=[
            lbann.CallbackPrintModelDescription(),
            lbann.CallbackPrint(),
            lbann.CallbackTimer(),
            lbann.CallbackDumpOutputs(),
        ],
    )

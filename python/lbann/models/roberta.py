import numpy as np
import math
import sys
import os

import lbann
import lbann.modules

ACT2FN = {
    "relu": lbann.Relu,
    "silu": lbann.modules.Silu,
    "swish": lbann.modules.Silu,
    "gelu": lbann.modules.Gelu,
    "tanh": lbann.Tanh,
    "sigmoid": lbann.Sigmoid,
}



def create_position_ids_from_input_ids(
    input_ids, input_shape, padding_idx, past_key_values_length=0
):
    padding_idx = lbann.Constant(value=padding_idx, num_neurons=input_shape)
    mask = lbann.NotEqual(input_ids, padding_idx)
    incremental_indices = lbann.Multiply(
        lbann.AddConstant(
            lbann.modules.Cumsum(mask, input_shape, axis=1),
            constant=past_key_values_length,
        ),
        mask,
    )
    incremental_indices = lbann.Add(incremental_indices, padding_idx)

    return incremental_indices


def _load_pretrained_weights(
    *fn,
    file_dir=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "pretrained_weights"
    ),
    load_weights=True,
):
    if not load_weights:
        return []

    # Use custom directory for loading weights
    if isinstance(load_weights, str):
        file_dir = load_weights

    weights = []
    for f in fn:
        w_file = os.path.join(file_dir, f + ".npy")
        if not os.path.isfile(w_file):
            raise ValueError(f"Pretrained weight file does not exist: {w_file}")
        weights.append(lbann.Weights(initializer=lbann.NumpyInitializer(file=w_file)))

    if len(weights) == 1:
        weights = weights[0]
    return weights

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
                value=0, num_neurons=self.input_shape
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

        embeddings = lbann.modules.PytorchLayerNorm(
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

    def create_position_ids_from_inputs_embeds(self, input_embeds):
        sequence_length = self.input_shape[1]
        position_ids = range(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1
        )
        position_ids = lbann.WeightsLayer(
            weights=lbann.Weights(
                initializer=lbann.ValueInitializer(values=position_ids),
                optimizer=lbann.NoOptimizer(),
            ),
            dims=[sequence_length],
        )
        position_ids = lbann.Reshape(position_ids, dims=[1, sequence_length])
        position_ids = lbann.Tessellate(
            position_ids, dims=self.input_shape[:-1]
        )
        return position_ids


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
        hidden_states = lbann.Reshape(hidden_states, dims=self.input_shape)
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
        x = lbann.Reshape(x, dims=new_x_shape)
        return lbann.modules.Permute(
            x, new_x_shape, axes=(0, 2, 1, 3), return_dims=True
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
    ):
        mixed_query_layer, query_shape = lbann.modules.PytorchLinear(
            hidden_states,
            self.input_shape,
            self.all_head_size,
            weights=_load_pretrained_weights(
                ".".join((self.name, "query.weight")),
                ".".join((self.name, "query.bias")),
                load_weights=self.load_weights,
            ),
            name=".".join((self.name, "query")),
            return_dims=True,
        )
        query_layer, query_shape = self.transpose_for_scores(
            mixed_query_layer, query_shape
        )

        key_layer, key_shape = lbann.modules.PytorchLinear(
            hidden_states,
            self.input_shape,
            self.all_head_size,
            weights=_load_pretrained_weights(
                ".".join((self.name, "key.weight")),
                ".".join((self.name, "key.bias")),
                load_weights=self.load_weights,
            ),
            name=".".join((self.name, "key")),
            return_dims=True,
        )
        key_layer, key_shape = self.transpose_for_scores(key_layer, key_shape)

        value_layer, value_shape = lbann.modules.PytorchLinear(
            hidden_states,
            self.input_shape,
            self.all_head_size,
            weights=_load_pretrained_weights(
                ".".join((self.name, "value.weight")),
                ".".join((self.name, "value.bias")),
                load_weights=self.load_weights,
            ),
            name=".".join((self.name, "value")),
            return_dims=True,
        )
        value_layer, value_shape = self.transpose_for_scores(value_layer, value_shape)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        key_layer, key_shape = lbann.modules.Permute(
            key_layer, key_shape, axes=(0, 1, -1, -2), return_dims=True
        )
        attention_scores, attention_shape = lbann.modules.PytorchMatmul(
            query_layer,
            query_shape,
            key_layer,
            key_shape,
            return_dims=True,
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
            dims=[np.prod(attention_shape[:-1]), attention_shape[-1]],
        )
        attention_probs = lbann.ChannelwiseSoftmax(attention_scores)
        attention_probs = lbann.Reshape(attention_probs, dims=attention_shape)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = lbann.Dropout(
            attention_probs,
            keep_prob=self.attention_probs_dropout_prob,
        )

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = lbann.Multiply(attention_probs, head_mask)

        context_layer, context_shape = lbann.modules.PytorchMatmul(
            attention_probs,
            attention_shape,
            value_layer,
            value_shape,
            return_dims=True,
        )
        context_layer, context_shape = lbann.modules.Permute(
            context_layer,
            context_shape,
            axes=(0, 2, 1, 3),
            return_dims=True,
        )
        new_context_layer_shape = context_shape[:-2] + (self.all_head_size,)
        context_layer = lbann.Reshape(context_layer, dims=self.input_shape)

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
        hidden_states, hidden_shape = lbann.modules.PytorchLinear(
            hidden_states,
            self.input_shape,
            self.hidden_size,
            weights=_load_pretrained_weights(
                ".".join((self.name, "dense.weight")),
                ".".join((self.name, "dense.bias")),
                load_weights=self.load_weights,
            ),
            name=".".join((self.name, "dense")),
            return_dims=True,
        )
        hidden_states = lbann.Dropout(hidden_states, keep_prob=self.hidden_dropout_prob)
        hidden_states = lbann.modules.PytorchLayerNorm(
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
        hidden_states = lbann.modules.PytorchLinear(
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
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


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
        hidden_states, hidden_shape = lbann.modules.PytorchLinear(
            hidden_states,
            self.input_shape,
            self.hidden_size,
            weights=_load_pretrained_weights(
                ".".join((self.name, "dense.weight")),
                ".".join((self.name, "dense.bias")),
                load_weights=self.load_weights,
            ),
            name=".".join((self.name, "dense")),
            return_dims=True,
        )
        hidden_states = lbann.Dropout(hidden_states, keep_prob=self.hidden_dropout_prob)
        hidden_states = lbann.modules.PytorchLayerNorm(
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
            hidden_states, axis=1, slice_points=[0, 1]
        )
        pooled_output = lbann.modules.PytorchLinear(
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


class RoBERTa(lbann.modules.Module):
    def __init__(self, config, add_pooling_layer=True, load_weights=True):
        self.config = config

        # A custom directory can be passed instead of True/False
        if isinstance(load_weights, str):
            if not os.path.isdir(load_weights):
                raise ValueError(
                    f"Path to pretrained weights does not exist: {load_weights}"
                )

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

    def extend_attention_mask(self, attention_mask):
        tmp_attn_shape = [
            self.attn_mask_shape[0],
            np.prod(self.attn_mask_shape[1:3]),
            self.attn_mask_shape[3],
        ]
        extended_attention_mask = lbann.Reshape(
            attention_mask, dims=[self.input_shape[0], 1, self.input_shape[1]]
        )
        extended_attention_mask = lbann.Reshape(
            lbann.Tessellate(extended_attention_mask, dims=tmp_attn_shape),
            dims=self.attn_mask_shape,
        )
        return extended_attention_mask

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
                value=1, num_neurons=self.input_shape
            )

        if token_type_ids is None:
            token_type_ids = lbann.Constant(
                value=0, num_neurons=self.input_shape
            )

        if head_mask is None:
            head_mask = [None] * self.config.num_hidden_layers

        extended_attention_mask = self.extend_attention_mask(attention_mask)

        input_ids = lbann.Reshape(input_ids, dims=self.input_shape)
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )
        encoder_output = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
        )
        pooled_output = self.pooler(encoder_output) if self.pooler is not None else None

        if pooled_output is not None:
            return pooled_output
        else:
            return encoder_output

class RobertaLMHead(lbann.modules.Module):
    """Roberta Head for masked language modeling."""

    def __init__(self, config, name,load_weights=True):
        self.config = config

        # A custom directory can be passed instead of True/False
        if isinstance(load_weights, str):
            if not os.path.isdir(load_weights):
                raise ValueError(f"Path to pretrained weights does not exist: {load_weights}")

        self.input_shape = config.input_shape + (config.hidden_size,)
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.hidden_dropout_prob = config.hidden_dropout_prob
        self.layer_norm_eps = config.layer_norm_eps
        self.name = name
        self.load_weights = load_weights
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
        
    def forward(self, input_tensor):
        
        #x = self.dense(features)
        hidden_states, hidden_shape = lbann.modules.PytorchLinear(
            input_tensor,
            self.input_shape,
            self.hidden_size,
            weights=_load_pretrained_weights(
                ".".join((self.name, "dense.weight")),
                ".".join((self.name, "dense.bias")),
                load_weights=self.load_weights,
            ),
            name=".".join((self.name, "dense")),
            return_dims=True,
        )
        
        #x = gelu(x)
        hidden_states = self.intermediate_act_fn(hidden_states)

        #x = self.layer_norm(x)
        hidden_states = lbann.modules.PytorchLayerNorm(
            lbann.Add(hidden_states, input_tensor),
            self.layer_norm_eps,
            hidden_shape,
            weights=_load_pretrained_weights(
                ".".join((self.name, "layer_norm.weightbias")),
                load_weights=self.load_weights,
            ),
            name=".".join((self.name, "layer_norm")),
        )

        #x = self.decoder(x)
        hidden_states, hidden_shape = lbann.modules.PytorchLinear(
            input_tensor,
            hidden_shape,
            self.vocab_size,
            weights=_load_pretrained_weights(
                ".".join((self.name, "decoder.weight")),
                ".".join((self.name, "decoder.bias")),
                load_weights=self.load_weights,
            ),
            name=".".join((self.name, "decoder")),
            return_dims=True,
        )

        return hidden_states

class RoBERTaMLM(lbann.modules.Module):
    def __init__(self, config, load_weights=True):

        # A custom directory can be passed instead of True/False
        if isinstance(load_weights, str):
            if not os.path.isdir(load_weights):
                raise ValueError(f"Path to pretrained weights does not exist: {load_weights}")        
                
        self.roberta = RoBERTa(config, add_pooling_layer=False, load_weights=load_weights)
        self.lm_head = RobertaLMHead(config, "lm_head",load_weights=load_weights)
        
    def forward(self,input_ids):
            
        output = self.roberta(input_ids)
        output = self.lm_head(output)
    
        return output

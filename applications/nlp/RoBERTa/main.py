import numpy as np
import math
import sys
import os

import lbann
import lbann.modules
from lbann.util import str_list

_Permute_cache = {}
_Cumsum_cache = {}


class RobertaEmbeddings(lbann.modules.Module):
    def __init__(self, config):
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

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        past_key_values_length=0,
    ):

        if position_ids is None:
            if input_ids is not None:
                position_ids = create_position_ids_from_input_ids(
                    input_ids,
                    self.input_shape,
                    self.padding_idx,
                    past_key_values_length,
                )
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(
                    inputs_embeds
                )

        if token_type_ids is None:
            # TODO
            if hasattr(self, "token_type_ids"):
                sys.exit("Not yet implemented")
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(
                    input_shape[0], seq_length
                )
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = lbann.Constant(
                    value=0, num_neurons=str_list(self.input_shape)
                )

        if inputs_embeds is None:
            inputs_embeds = lbann.Embedding(
                input_ids,
                num_embeddings=self.vocab_size,
                embedding_dim=self.hidden_size,
                padding_idx=self.pad_token_id,
                weights=_load_pretrained_weights("embeddings.word_embeddings.weight"),
            )
        token_type_embeddings = lbann.Embedding(
            token_type_ids,
            num_embeddings=self.type_vocab_size,
            embedding_dim=self.hidden_size,
            weights=_load_pretrained_weights("embeddings.token_type_embeddings.weight"),
        )

        embeddings = lbann.Add(inputs_embeds, token_type_embeddings)
        if self.position_embedding_type == "absolute":
            position_embeddings = lbann.Embedding(
                position_ids,
                num_embeddings=self.max_position_embeddings,
                embedding_dim=self.hidden_size,
                padding_idx=self.pad_token_id,
                weights=_load_pretrained_weights(
                    "embeddings.position_embeddings.weight"
                ),
            )
            embeddings = lbann.Add(embeddings, position_embeddings)

        embeddings = _LayerNorm(
            embeddings,
            self.layer_norm_eps,
            self.input_shape + (self.hidden_size,),
            weights=_load_pretrained_weights("embeddings.layernorm.weightbias"),
        )
        embeddings = lbann.Dropout(embeddings, keep_prob=self.hidden_dropout_prob)
        return embeddings


class RobertaEncoder(lbann.modules.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_shape = config.input_shape + (config.hidden_size,)
        self.layer = [
            RobertaLayer(config, layer_id=i) for i in range(config.num_hidden_layers)
        ]

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = (
            () if output_attentions and self.config.add_cross_attention else None
        )

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if getattr(self.config, "gradient_checkpointing", False) and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                        "`use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        # TODO
        # return BaseModelOutputWithPastAndCrossAttentions(
        #    last_hidden_state=hidden_states,
        #    past_key_values=next_decoder_cache,
        #    hidden_states=all_hidden_states,
        #    attentions=all_self_attentions,
        #    cross_attentions=all_cross_attentions,
        # )
        return (
            hidden_states,
            next_decoder_cache,
            all_hidden_states,
            all_self_attentions,
            all_cross_attentions,
        )


class RobertaLayer(lbann.modules.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.input_shape = config.input_shape + (config.hidden_size,)
        self.attention = RobertaAttention(config, layer_id)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        self.layer_id = layer_id
        if self.add_cross_attention:
            assert (
                self.is_decoder
            ), f"{self} should be used as a decoder model if cross attention is added"
            self.crossattention = RobertaAttention(config)
        self.intermediate = RobertaIntermediate(config, layer_id)
        self.output = RobertaOutput(config, layer_id)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = (
            past_key_value[:2] if past_key_value is not None else None
        )
        hidden_states = lbann.Reshape(hidden_states, dims=str_list(self.input_shape))
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[
                1:
            ]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            assert hasattr(
                self, "crossattention"
            ), f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = (
                past_key_value[-2:] if past_key_value is not None else None
            )
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = (
                outputs + cross_attention_outputs[1:-1]
            )  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = self.feed_forward_chunk(attention_output)
        outputs = (layer_output,) + outputs  # to be removed

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class RobertaSelfAttention(lbann.modules.Module):
    def __init__(self, config, layer_id):
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
        self.is_decoder = config.is_decoder
        self.layer_id = layer_id

    def transpose_for_scores(self, x, dims):
        new_x_shape = dims[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = lbann.Reshape(x, dims=lbann.util.str_list(new_x_shape))
        return _Permute(x, new_x_shape, axes=(0, 2, 1, 3))

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        mixed_query_layer, query_shape = _Linear(
            hidden_states,
            self.input_shape,
            self.all_head_size,
            weights=(
                _load_pretrained_weights(
                    f"encoder.layer.{self.layer_id}.attention.self.query.weight"
                ),
                _load_pretrained_weights(
                    f"encoder.layer.{self.layer_id}.attention.self.query.bias"
                ),
            ),
        )

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        # TODO
        if is_cross_attention and past_key_value is not None:
            sys.exit("Not yet implemented")
        elif is_cross_attention:
            sys.exit("Not yet implemented")
        elif past_key_value is not None:
            sys.exit("Not yet implemented")
        else:
            key_layer, key_shape = _Linear(
                hidden_states,
                self.input_shape,
                self.all_head_size,
                weights=(
                    _load_pretrained_weights(
                        f"encoder.layer.{self.layer_id}.attention.self.key.weight"
                    ),
                    _load_pretrained_weights(
                        f"encoder.layer.{self.layer_id}.attention.self.key.bias"
                    ),
                ),
            )
            key_layer, key_shape = self.transpose_for_scores(key_layer, key_shape)
            value_layer, value_shape = _Linear(
                hidden_states,
                self.input_shape,
                self.all_head_size,
                weights=(
                    _load_pretrained_weights(
                        f"encoder.layer.{self.layer_id}.attention.self.value.weight"
                    ),
                    _load_pretrained_weights(
                        f"encoder.layer.{self.layer_id}.attention.self.value.bias"
                    ),
                ),
            )
            value_layer, value_shape = self.transpose_for_scores(
                value_layer, value_shape
            )

        query_layer, query_shape = self.transpose_for_scores(
            mixed_query_layer, query_shape
        )

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        key_layer, key_shape = _Permute(key_layer, key_shape, axes=(0, 1, -1, -2))
        attention_scores, attention_shape = _Matmul(
            query_layer, query_shape, key_layer, key_shape
        )

        denom = lbann.Constant(
            value=math.sqrt(self.attention_head_size),
            num_neurons=str_list(attention_shape),
        )
        attention_score = lbann.Divide(attention_scores, denom)
        # TODO
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in RobertaModel forward() function)
            attention_scores = lbann.Add(attention_scores, attention_mask)

        # Normalize the attention scores to probabilities.
        attention_probs = lbann.ChannelwiseSoftmax(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = lbann.Dropout(
            attention_probs, keep_prob=self.attention_probs_dropout_prob
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

        outputs = (
            (context_layer, attention_probs) if output_attentions else (context_layer,)
        )

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


class RobertaSelfOutput(lbann.modules.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.input_shape = config.input_shape + (config.hidden_size,)
        self.hidden_size = config.hidden_size
        self.layer_norm_eps = config.layer_norm_eps
        self.hidden_dropout_prob = config.hidden_dropout_prob
        self.layer_id = layer_id

    def forward(self, hidden_states, input_tensor):
        hidden_states, _ = _Linear(hidden_states, self.input_shape, self.hidden_size)
        hidden_states, hidden_shape = _Linear(
            hidden_states,
            self.input_shape,
            self.hidden_size,
            weights=(
                _load_pretrained_weights(
                    f"encoder.layer.{self.layer_id}.attention.output.dense.weight"
                ),
                _load_pretrained_weights(
                    f"encoder.layer.{self.layer_id}.attention.output.dense.bias"
                ),
            ),
        )
        hidden_states = lbann.Dropout(hidden_states, keep_prob=self.hidden_dropout_prob)
        hidden_states = _LayerNorm(
            lbann.Add(hidden_states, input_tensor),
            self.layer_norm_eps,
            hidden_shape,
            weights=_load_pretrained_weights(
                f"encoder.layer.{self.layer_id}.attention.output.layernorm.weightbias"
            ),
        )
        return hidden_states


class RobertaAttention(lbann.modules.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.input_shape = config.input_shape + (config.hidden_size,)
        self.self = RobertaSelfAttention(config, layer_id)
        self.output = RobertaSelfOutput(config, layer_id)
        self.pruned_heads = set()
        self.layer_id = layer_id

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + (
            self_outputs[1:],
        )  # add attentions if we output them
        return outputs


class RobertaIntermediate(lbann.modules.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.intermediate_size = config.intermediate_size
        self.input_shape = config.input_shape + (config.hidden_size,)
        self.layer_id = layer_id
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states, hidden_shape = _Linear(
            hidden_states,
            self.input_shape,
            self.intermediate_size,
            weights=(
                _load_pretrained_weights(
                    f"encoder.layer.{self.layer_id}.intermediate.dense.weight"
                ),
                _load_pretrained_weights(
                    f"encoder.layer.{self.layer_id}.intermediate.dense.bias"
                ),
            ),
        )
        hidden_states = self.intermediate_act_fn(hidden_states, hidden_shape)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertOutput
class RobertaOutput(lbann.modules.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.input_shape = config.input_shape + (config.intermediate_size,)
        self.hidden_size = config.hidden_size
        self.hidden_dropout_prob = config.hidden_dropout_prob
        self.layer_norm_eps = config.layer_norm_eps
        self.layer_id = layer_id

    def forward(self, hidden_states, input_tensor):
        hidden_states, hidden_shape = _Linear(
            hidden_states,
            self.input_shape,
            self.hidden_size,
            weights=(
                _load_pretrained_weights(
                    f"encoder.layer.{self.layer_id}.output.dense.weight"
                ),
                _load_pretrained_weights(
                    f"encoder.layer.{self.layer_id}.output.dense.bias"
                ),
            ),
        )
        hidden_states = lbann.Dropout(hidden_states, keep_prob=self.hidden_dropout_prob)
        hidden_states = lbann.Add(hidden_states, input_tensor)
        hidden_states = _LayerNorm(
            hidden_states,
            self.layer_norm_eps,
            hidden_shape,
            weights=_load_pretrained_weights(
                f"encoder.layer.{self.layer_id}.output.layernorm.weightbias"
            ),
        )
        return hidden_states


class RobertaPooler(lbann.modules.Module):
    def __init__(self, config):
        super().__init__()
        self.input_shape = config.input_shape + (config.hidden_size,)
        self.hidden_size = config.hidden_size

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = lbann.Slice(
            hidden_states, axis=0, slice_points=str_list([0, 1])
        )
        pooled_output, pooled_output_shape = _Linear(
            first_token_tensor,
            (1,) + self.input_shape[1:],
            self.hidden_size,
            weights=(
                _load_pretrained_weights("pooler.dense.weight"),
                _load_pretrained_weights("pooler.dense.bias"),
            ),
        )
        pooled_output = lbann.Tanh(pooled_output)
        return pooled_output


class RobertaModel(lbann.modules.Module):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    # Copied from transformers.models.bert.modeling_bert.BertModel.__init__ with Bert->Roberta
    def __init__(self, config, add_pooling_layer=True):
        # super().__init__(config)
        self.config = config

        self.embeddings = RobertaEmbeddings(config)
        self.encoder = RobertaEncoder(config)

        self.pooler = RobertaPooler(config) if add_pooling_layer else None
        self.input_shape = config.input_shape

        # self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        # if input_ids is not None and inputs_embeds is not None:
        #  raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        # elif input_ids is not None:
        #  input_shape = input_ids.size()
        #  batch_size, seq_length = input_shape
        # elif inputs_embeds is not None:
        #  input_shape = inputs_embeds.size()[:-1]
        #  batch_size, seq_length = input_shape
        # else:
        #  raise ValueError("You have to specify either input_ids or inputs_embeds")

        # past_key_values_length
        past_key_values_length = (
            past_key_values[0][0].shape[2] if past_key_values is not None else 0
        )

        if attention_mask is None:
            attention_mask = lbann.Constant(
                value=1, num_neurons=str_list(self.input_shape)
            )

        if token_type_ids is None:
            token_type_ids = lbann.Constant(
                value=0, num_neurons=str_list(self.input_shape)
            )

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        # extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            (
                encoder_batch_size,
                encoder_sequence_length,
                _,
            ) = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(
                encoder_attention_mask
            )
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        # head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        head_mask = [None] * self.config.num_hidden_layers

        input_ids = lbann.Reshape(input_ids, dims=str_list(self.input_shape))
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        embedding_output = lbann.Reshape(
            embedding_output, dims=str_list(self.input_shape + (768,))
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[2][0]
        pooled_output = (
            self.pooler(sequence_output) if self.pooler is not None else None
        )

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        # TODO
        # return BaseModelOutputWithPoolingAndCrossAttentions(
        #  last_hidden_state=sequence_output,
        #  pooler_output=pooled_output,
        #  past_key_values=encoder_outputs.past_key_values,
        #  hidden_states=encoder_outputs.hidden_states,
        #  attentions=encoder_outputs.attentions,
        #  cross_attentions=encoder_outputs.cross_attentions,
        # )
        return (
            sequence_output,
            pooled_output,
            encoder_outputs[1],
            encoder_outputs[2],
            encoder_outputs[3],
            encoder_outputs[4],
        )


def _relu(x, _):
    return lbann.Relu(x)


def _silu(x, _):
    return lbann.Multiply(x, lbann.Sigmoid(x))


def _gelu(x, x_shape):
    # return 0.5 * x * (1 + tanh(sqrt(pi / 2) * (x + 0.044715 * x ** 3)))
    # Based on: https://github.com/pytorch/pytorch/issues/20464#issuecomment-492339005
    sqrt_pi_over_2 = math.sqrt(math.pi / 2)
    b_coef = 0.044715
    x_cubed = lbann.Multiply(lbann.Multiply(lbann.Identity(x), x), x)
    inner_tanh_x_comp = lbann.Add(x, lbann.Scale(x_cubed, constant=b_coef))
    tanh_x = lbann.Tanh(lbann.Scale(inner_tanh_x_comp, constant=sqrt_pi_over_2))
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


def _Permute(x, dims, axes=None):
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
    y = lbann.Reshape(y, dims=str_list(list(new_dims)))
    return y, tuple(new_dims)


def _Cumsum(x, dims, axis=0):
    global _Cumsum_cache

    if len(dims) != 2:
        sys.exit("dims > 2 not tested/supported for cumsum")
    if (axis < 0) or (axis > 1):
        sys.exit("Unsupported cumsum axis: {}".format(axis))
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
    fn, file_dir="/usr/workspace/wyatt5/pascal/sbert/authortransformer/npy_weights"
):
    weights_file = os.path.join(file_dir, fn + ".npy")
    dims = np.load(weights_file).shape
    weights = _load_pretrained_weights(fn, file_dir)
    weights = lbann.WeightsLayer(weights=weights, dims=str_list(dims))
    return weights, dims


def _load_pretrained_weights(
    fn, file_dir="/usr/workspace/wyatt5/pascal/sbert/authortransformer/npy_weights"
):
    weights_file = os.path.join(file_dir, fn + ".npy")
    print("loading weight:", fn)
    weights = lbann.Weights(initializer=lbann.NumpyInitializer(file=weights_file))
    return weights


# Mimics torch.matmul in LBANN
def _Matmul(x, x_shape, y, y_shape):
    if len(x_shape) != len(y_shape):
        sys.exit("Broadcasting not fully implemented, tensors must have same dimension")
    need_reshape = (len(x_shape) > 3) and (len(y_shape) > 3)
    if need_reshape:
        if x_shape[:-2] != y_shape[:-2]:
            sys.exit("The first n-2 dimensions must match")
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
def _Linear(x, input_shape, hidden_size, weights=None):
    need_reshape = len(input_shape) > 2
    if need_reshape:
        new_in_shape = (np.prod(input_shape[:-1]), input_shape[-1])
        x = lbann.Reshape(x, dims=str_list(new_in_shape))

    if weights is not None:
        y = lbann.ChannelwiseFullyConnected(
            x,
            output_channel_dims=[hidden_size],
            weights=weights,
        )
    else:
        y = lbann.ChannelwiseFullyConnected(
            x,
            output_channel_dims=[hidden_size],
        )

    if need_reshape:
        new_out_shape = input_shape[:-1] + (hidden_size,)
        y = lbann.Reshape(y, dims=str_list(new_out_shape))
    else:
        new_out_shape = (input_shape[0], hidden_size)

    return y, new_out_shape


# Mimics torch.nn.layernorm in LBANN
def _LayerNorm(x, epsilon, input_shape=None, weights=None):
    x = lbann.LayerNorm(x, epsilon=epsilon)
    if weights is not None:
        if input_shape is None:
            sys.exit("input shape must be provided for loading weights on LayerNorm")

        x, new_x_shape = _Permute(x, input_shape)
        x = lbann.ChannelwiseScaleBias(x, weights=weights)
        x, _ = _Permute(x, new_x_shape)

    return x


def make_model(config):

    # Input data
    input_ = lbann.Slice(
        lbann.Input(data_field="samples"), slice_points=str_list([0, 1, 1 + 16 * 32])
    )
    labels = lbann.Identity(input_)
    sample = lbann.Reshape(input_, dims=str_list([16, 32]))

    # Load pretrained weights
    position_ids, _ = _load_pretrained_weights_layer("embeddings.position_ids")

    # Model
    model = RobertaModel(config)
    out = model(sample, position_ids=position_ids)
    out = lbann.FullyConnected(out[1], num_neurons=1)
    probs = lbann.Softmax(out)

    loss = lbann.CrossEntropy(probs, labels)
    acc = lbann.CategoricalAccuracy(probs, labels)

    layer_list = list(lbann.traverse_layer_graph([sample, labels]))

    metrics = [lbann.Metric(acc, name="accuracy")]

    return lbann.Model(
        10,
        layers=lbann.traverse_layer_graph(input_),
        objective_function=loss,
        metrics=metrics,
        callbacks=[
            lbann.CallbackPrintModelDescription(),
            lbann.CallbackPrint(),
            lbann.CallbackTimer(),
        ],
    )

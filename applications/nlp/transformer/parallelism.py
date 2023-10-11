"""
Different model distributions that showcase how LBANN can distribute transformer
training. Use the flags in any of the driver scripts to invoke the parallelism
strategies found in this file.
"""
import argparse
import itertools
import lbann
import lbann.models.subgraph.transformer
from typing import Any, Dict, Optional, List, Tuple, Union

#############################################################################


# Fully-sharded data parallelism
def apply_fsdp(module: lbann.models.Transformer, args: argparse.Namespace):
    """
    Applies a sharded-weight data-parallel strategy on every weight or
    MLP weights only.

    :param module: Transformer module to modify.
    :param args: Command-line arguments.
    """
    if not args.fsdp:
        return
    if args.ffn_parallel:
        raise ValueError('FSDP is incompatible with model parallelism')

    # Go over all encoders and decoders
    for i, submodule in itertools.chain(enumerate(module.encoder),
                                        enumerate(module.decoder)):
        for w in submodule.fc1_weights:
            w.sharded = True
        for w in submodule.fc2_weights:
            w.sharded = True


# Model (FFN tensor) parallelism
def apply_ffn_model_parallelism(module: lbann.models.Transformer,
                                args: argparse.Namespace,
                                layers: Optional[List[int]] = None):
    """
    Applies a model-parallel strategy on the MLP in each transformer block, or
    on a given list of transformer blocks, if requested.

    :param module: Transformer module to modify.
    :param args: Command-line arguments.
    :param layers: If not None, a list of integers representing which blocks
                   to apply model parallelism to.
    """
    if not args.ffn_parallel:
        return

    # Go over all encoders and decoders
    for i, submodule in itertools.chain(enumerate(module.encoder),
                                        enumerate(module.decoder)):
        if layers is not None and i not in layers:
            continue

        # Apply model parallelism
        submodule.extra_ffn_args = dict(data_layout='model_parallel')


# LM head parallelism
def apply_lm_head_model_parallelism(lm_head: lbann.Layer,
                                    args: argparse.Namespace):
    """
    Applies a model-parallel strategy on the embedding of the language modeling
    head of the transformer, if requested.

    :param lm_head: The layer from which to start applying model parallelism
                    (will apply until loss).
    :param args: Command-line arguments.
    """
    if not args.lm_head_parallel:
        return

    # Find all layers starting from LM head onwards
    layers = list(lbann.traverse_layer_graph(lm_head))
    head_layers = layers[layers.index(lm_head):]

    # Apply model parallelism
    for layer in head_layers:
        layer.data_layout = 'model_parallel'

    print('Applied model parallelism on the following layers: '
          f'{", ".join(layer.name for layer in head_layers)}')


#############################################################################
# Subgraph (attention head) parallelism


def apply_subgraph_parallelism(
    module: lbann.models.Transformer, args: argparse.Namespace
) -> Tuple[Union[lbann.models.Transformer, lbann.models.subgraph.transformer.
                 TransformerAllSubgraph], Dict[str, Any]]:
    """
    Applies a subgraph parallelism strategy on the attention heads in each
    transformer block, if requested.

    :param module: Transformer module to use.
    :param args: Command-line arguments.
    :return: A 2-tuple of (module, extra_model_args), where module is either a
             ``Transformer`` module, or ``TransformerAllSubgraph`` with
             all-subgraph (i.e., MLP per branch) parallelism.
             The second tuple entry is a set of extra arguments to pass to
             the ``lbann.Model`` upon creation.
    """
    if args.attn_head_parallel == 0:  # Not requested
        return module, {}

    # Create new module from existing one
    if args.subgraph_parallelism_all:
        sgmodule = lbann.models.subgraph.TransformerAllSubgraph(
            # Transfomer parameters
            hidden_size=module.hidden_size,
            num_heads=module.num_heads,
            num_encoder_layers=len(module.encoder),
            num_decoder_layers=len(module.decoder),
            filter_size=module.feedforward_size,
            dropout=module.dropout,
            attn_dropout=module.attn_dropout,
            pre_layernorm=module.pre_layernorm,
            activation=module.activation,
            name=module.name,
            # Subgraph parameters
            branches=args.attn_head_parallel,
            ENABLE_ALLSUBGRAPH=True,
            ENABLE_Concat=args.subgraph_parallelism_concat,
        )
    else:
        sgmodule = lbann.models.Transformer(
            # Transfomer parameters
            hidden_size=module.hidden_size,
            num_heads=module.num_heads,
            num_encoder_layers=len(module.encoder),
            num_decoder_layers=len(module.decoder),
            feedforward_size=module.feedforward_size,
            dropout=module.dropout,
            attn_dropout=module.attn_dropout,
            pre_layernorm=module.pre_layernorm,
            activation=module.activation,
            name=module.name,
            # Subgraph parameters
            parallel_attention_heads=args.attn_head_parallel,
        )

    # Add model arguments
    extra_model_kwargs = dict(
        subgraph_communication=lbann.SubgraphCommunication.COLL_OPT,
        subgraph_topology=args.head_parallel_subgraph_topology,
        subgraph_num_common_resources=args.head_parallel_parent_resources,
    )

    return sgmodule, extra_model_kwargs


def add_transformer_parallelism_arguments(parser: argparse.Namespace,
                                          subgraph: bool = True):

    #######################################
    # Model parallelism
    parser.add_argument(
        '--ffn-parallel',
        action='store_true',
        help=
        'Enable model parallelism on the feedforward part of the transformer '
        'blocks')

    parser.add_argument(
        '--lm-head-parallel',
        action='store_true',
        help='Enable model parallelism on the language modeling head')

    #######################################
    # Subgraph (attention head) parallelism
    if not subgraph:
        return

    parser.add_argument('--attn-head-parallel',
                        action='store',
                        default=0,
                        type=int,
                        help='Enable subgraph parallelism on attention heads, '
                        'defining the degree of parallelism. 0 disables '
                        '(default: 0)',
                        metavar='NUM')

    parser.add_argument(
        '--head-parallel-subgraph-topology',
        action='store',
        default=0,
        type=int,
        help='Stategy for topology aware subgraph parallelism on attention '
        'heads (default: 0) ',
        metavar='NUM')

    parser.add_argument(
        '--head-parallel-parent-resources',
        action='store',
        default=0,
        type=int,
        help='Number of resources for parent/common layers for attention head '
        'subgraph parallelism (corresponds to use all ranks) (default: 0)',
        metavar='NUM')

    parser.add_argument(
        '--subgraph-parallelism-all',
        action='store_true',
        help='Enable subgraph parallelism for all layers in transformer, '
        'including decomposing the MLP into distinct subgraphs')

    parser.add_argument(
        '--subgraph-parallelism-concat',
        action='store_true',
        help=
        'Apply concat operation after each transformer block (only applies if '
        '--subgraph-parallelism-all is given)')

    parser.add_argument(
        '--fsdp',
        action='store_true',
        help='Apply Fully-Sharded Data-Parallelism (FSDP) and shard all weights'
    )

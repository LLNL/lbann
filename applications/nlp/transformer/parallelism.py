"""
Different model distributions that showcase how LBANN can distribute transformer
training. Use the flags in any of the driver scripts to invoke the parallelism
strategies found in this file.
"""
import argparse
import itertools
import lbann
import lbann.models.subgraph.transformer
import math
import re
from typing import Any, Dict, Optional, List, Tuple, Union

#############################################################################

def _get_sharding_strategy(args: argparse.Namespace) -> lbann.ShardingStrategy:
    if args.fsdp_ranks > 0:
        return lbann.ShardingStrategy.GRID_ROWS
    return lbann.ShardingStrategy.FULL


# Fully-sharded data parallelism (MLP only)
def apply_fsdp_mlp(module: lbann.models.Transformer,
                   other_weights: List[lbann.Weights],
                   args: argparse.Namespace):
    """
    Applies a sharded-weight data-parallel strategy on every weight or
    MLP weights only.

    :param module: Transformer module to modify.
    :param other_weights: List of other weights to shard (e.g., embeddings).
    :param args: Command-line arguments.
    """
    if not args.fsdp_mlp:
        return
    if args.ffn_parallel:
        raise ValueError('FSDP is incompatible with model parallelism')

    # Loop over all encoders and decoders
    for _, submodule in itertools.chain(enumerate(module.encoder),
                                        enumerate(module.decoder)):
        for w in submodule.fc1_weights:
            w.sharded = True
            w.sharding_strategy = _get_sharding_strategy(args)
        for w in submodule.fc2_weights:
            w.sharded = True
            w.sharding_strategy = _get_sharding_strategy(args)

    for w in other_weights:
        w.sharded = True
        w.sharding_strategy = _get_sharding_strategy(args)


# Fully-sharded data parallelism (all weights)
def apply_fsdp_allweights(model: lbann.Model, args: argparse.Namespace):
    """
    Applies a sharded-weight data-parallel strategy on every weight.

    :param model: LBANN model to use.
    :param args: Command-line arguments.
    """
    if not args.fsdp:
        return
    if args.ffn_parallel:
        raise ValueError('FSDP is incompatible with model parallelism')

    # Loop over all weights
    for layer in model.layers:
        # As a heuristic, only shard the first set of weights (i.e., no
        # biases) and skip layer normalization
        if 'LayerNorm' in str(type(layer)):
            continue
        if layer.weights:
            if len(layer.weights) > 0:
                layer.weights[0].sharded = True
                layer.weights[0].sharding_strategy = _get_sharding_strategy(args)


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


#############################################################################
# Layer parallelism

lp_grids = None
def apply_layer_parallelism(module: lbann.models.Transformer,
                            model: lbann.Model, args: argparse.Namespace):
    """
    Applies a model-parallel strategy on sequences of contiguous transformer
    blocks, sometimes referred to as pipeline parallelism or layer parallelism.

    :param module: Transformer module to take as reference for block counts.
    :param model: The model to modify.
    :param args: Command-line arguments.
    :param layers: If not None, a list of integers representing which blocks
                   to apply model parallelism to.
    """
    if not args.layer_parallel:
        return

    lp_count = args.lp_count
    if args.lp_count == 0:
        lp_count = args.nodes * args.procs_per_node

    blocks = len(module.encoder) + len(module.decoder)

    # Assign blocks to increasing grid tags
    blocks_per_grid_tag = math.ceil(blocks / lp_count)
    cur_grid_tag = 0

    # Go over all layers in traversal order, applying grid tags in increasing order
    last_block_id = -1
    block_id = -1
    total_block_id = 0
    for layer in model.layers:
        if layer.name.startswith('transformer_decoder'):
            block_id = int(
                re.search(r'transformer_decoder(\d+)_',
                          layer.name).groups(1)[0])
        elif layer.name.startswith('transformer_encoder'):
            block_id = int(
                re.search(r'transformer_encoder(\d+)_',
                          layer.name).groups(1)[0])
        if last_block_id != block_id:
            if total_block_id % blocks_per_grid_tag == 0:
                cur_grid_tag += 1
            last_block_id = block_id
            total_block_id += 1

        # Apply layer parallelism
        layer.grid_tag = { 'value': cur_grid_tag }

    global lp_grids
    lp_grids = cur_grid_tag

def get_layer_parallel_args() -> List[str]:
    if lp_grids is not None:
        return ['--num-subgrids', str(lp_grids)]

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

    parser.add_argument(
        '--fsdp-ranks',
        default=0,
        type=int,
        help='Number of consecutive nodes to shard weights in FSDP. This '
        'setting will modify the LBANN process grid height. (default: 0, shard '
        'across all ranks)'
    )

    parser.add_argument(
        '--fsdp-mlp',
        action='store_true',
        help='Apply Fully-Sharded Data-Parallelism (FSDP) and shard MLP weights'
    )

    #######################################
    # Layer parallelism
    parser.add_argument(
        '--layer-parallel',
        action='store_true',
        help='Apply layer parallelism (also referred to as pipelining)')
    parser.add_argument(
        '--lp-count',
        default=0,
        type=int,
        help='In layer parallelism, the number of portions to divide network to'
        ' (Default: divide evenly between all ranks)')

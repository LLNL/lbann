"""
Different model distributions that showcase how LBANN can distribute transformer
training. Use the flags in any of the driver scripts to invoke the parallelism
strategies found in this file.
"""

##############
# Model (FFN tensor) parallelism

# LM head parallelism

##############
# Subgraph (attention head) parallelism

# Changes to model:
# transformer = lbann.models.subgraph.TransformerSubGraph(
#     branches = branches,
#     d_kv = d_kv,
#     ENABLE_ALLSUBGRAPH = ENABLE_ALLSUBGRAPH,
#     ENABLE_Concat = ENABLE_Concat,
#     ...
# )
# return lbann.Model(
#     subgraph_communication=lbann.SubgraphCommunication.COLL_OPT,
#     subgraph_topology=subgraph_topology,
#     subgraph_num_common_resources = subgraph_num_common_resources,
#     ...
# )

# Changes to main:
# parser.add_argument(
#     '--branches', action='store', default=0, type=int,
#     help='Number of Branches 0 means DP (default: 0)', metavar='NUM')

# parser.add_argument(
#     '--subgraph-topology', action='store', default=0, type=int,
#     help='Stategy for topology aware subgraph parallelism (default: 0) ', metavar='NUM')

# parser.add_argument(
#     '--subgraph-parent-resources', action='store', default=0, type=int,
#     help='NUmber of resources for parent/common layers (corresponds to use all ranks) (default: 0) ', metavar='NUM')

# parser.add_argument('--enable-allsubgraph', dest='ENABLE_ALLSUBGRAPH', action='store_true',
#                         help='Enable subgraph parallelism for common layers in Encoder')
# parser.add_argument('--enable-concat', dest='ENABLE_Concat', action='store_true',
#                         help='Apply concat operation after each encoder layers when AllSubgraph variable is given')

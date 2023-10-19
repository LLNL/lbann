import lbann
from lbann.modules import Module, NNConv
from lbann.util import str_list
import lbann.modules as nn
import math
import warnings


class Sequential(Module):
    """Sequential container for LBANN layers. Similar to:
       https://pytorch.org/docs/stable/generated/\
       torch.nn.Sequential.html#torch.nn.Sequential

       Only supports layers in lbann.Module. Need to think up a kwargs
       trick to make it usable for all layers.
    """
    def __init__(self, sequential_layer_list):
        super(Sequential, self).__init__()
        self.layers = sequential_layer_list

    def forward(self, x):
        temp = x
        for layer in self.layers:
            temp = layer(temp)
        return temp


class global_add_pool(Module):
    """Combines the node feature matrix into a single vector with 
        addition along the column axis
        
    """
    global_count = 0

    def __init__(self,
                 mask=None,
                 num_nodes=None,
                 name=None):
        """

            params:
                mask (Layer): (default: None)
                num_nodes (int): (default : None)
                name (string): (default: None)
        """
        super().__init__()
        global_add_pool.global_count += 1
        self.name = (name if name else
                     "global_add_pool")
        if mask is None:
            if num_nodes is None:
                ValueError("Either requires one of mask or num_nodes must be set")
            self.reduction = lbann.Constant(value=1,
                                            num_neurons=str_list([1, num_nodes]),
                                            name=self.name)
        else:
            if num_nodes is None:
                warnings.warn("Only one of mask or num_nodes should be set. Using mask value")
            self.reduction = mask 

    def forward(self, x):
        return lbann.MatMul(self.reduction, x, name=self.name + "_matmul")


class Graph_Conv(Module):
    global_count = 0
    """The customized gated graph convolution operator from the `"Gated Graph Sequence
    Neural Networks" <https://arxiv.org/abs/1511.05493>`_ paper for the FAST model descibed here: 
    https://github.com/LLNL/FAST/blob/5b8c2d775a67ad1716e26938492f1f5d037962a3/model/sgcnn/src/sgcnn/ggcnn.py#L22
    """
    def __init__(self,
                 num_nodes,
                 num_edges,
                 input_feature_dim,
                 output_feature_dim,
                 edge_feature_dim,
                 num_layers,
                 edge_nn=None,
                 name=None):
        """ Constructor for Graph_Conv object 
            Args:
                num_nodes (int)
                num_edges (int)
                input_feature_dim (int)
                output_feature_dim (int)
        """
        super(Graph_Conv, self).__init__()
        Graph_Conv.global_count += 1
        self.output_channel_dim = output_feature_dim
        self.input_feature_dim = input_feature_dim
        self.num_layers = num_layers
        self.num_nodes = num_nodes
        self.num_edges = num_edges

        self.edge_conv = NNConv(sequential_nn=edge_nn,
                                num_nodes=num_nodes,
                                num_edges=num_edges,
                                input_channels=output_feature_dim,
                                output_channels=output_feature_dim,
                                edge_input_channels=edge_feature_dim)

        self.name = (name if name else
                     'GraphConv_{}'.format(Graph_Conv.global_count))

        self.rnn = nn.ChannelwiseGRU(self.output_channel_dim, self.num_nodes)

        self.weights = []

        for i in range(num_layers):
            weight_init = \
                lbann.Weights(initializer=lbann.UniformInitializer(min=-1 / (math.sqrt(self.output_channel_dim)), 
                                                                   max=1 / (math.sqrt(self.output_channel_dim))))

            self.weights.append(nn.ChannelwiseFullyConnectedModule(self.output_channel_dim, 
                                                                   bias=False, 
                                                                   weights=weight_init,
                                                                   name=f"{self.name}_{i}_weight"))

    def forward(self,
                node_features,
                edge_features,
                edge_source_indices,
                edge_target_indices):
        """Apply gated graph convolution on the graph data

            Args: 
                node_features (Layer): A 2D matrix with dimensions (num_nodes, ) 
                edge_features (Layer): 
                edge_source_indices (Layer): 
                edge_target_indices (Layer):
        """
        if (self.input_feature_dim < self.output_channel_dim):

            num_zeros = self.output_channel_dim - self.input_feature_dim

            zeros = lbann.Constant(value=0,
                                   num_neurons=str_list([self.num_nodes, num_zeros]))

            node_features = lbann.Concatenation(node_features, zeros, axis=1)
      
        for layer in range(self.num_layers):
            nf_clone = lbann.Identity(lbann.Split(node_features))
           
            messages = self.weights[layer](nf_clone)

            aggregate = lbann.Scatter(messages, edge_source_indices,
                                      dims=str_list([self.num_nodes, self.output_channel_dim]))

            Node_FT_GRU_input = aggregate

            # Update node_features according to edge convolutions
            # Generate the neighbor matrices with gather 

            neighbor_features = lbann.Reshape(lbann.Gather(node_features, edge_source_indices),
                                              dims=str_list([self.num_edges, 1, self.output_channel_dim]),
                                              name=f"{self.name}_{layer}_reshape")
            Node_FT_GRU_PrevState = self.edge_conv(node_features,
                                                   neighbor_features,
                                                   edge_features,
                                                   edge_target_indices)

            node_features, _ = self.rnn(Node_FT_GRU_input, Node_FT_GRU_PrevState)
        
        return node_features


class Graph_Attention(Module):
    """docstring for Graph_Attention"""
    global_count = 0

    def __init__(self,
                 feat_size,
                 output_size,
                 name=None):
        super(Graph_Attention, self).__init__()
        Graph_Attention.global_count += 1
        self.nn_1 = Sequential([nn.ChannelwiseFullyConnectedModule(feat_size),
                                lbann.Softsign,
                                nn.ChannelwiseFullyConnectedModule(output_size),
                                lbann.Softsign
                                ])
        self.nn_2 = nn.ChannelwiseFullyConnectedModule(output_size,
                                                       activation=lbann.Softsign)
        self.name = (name if name else
                     'GraphAttention_{}'.format(Graph_Attention.global_count))

    def forward(self,
                updated_nodes,
                original_nodes):
        """
        Args:
            updated_nodes (Layer): 
            original_nodes (Layer): 
        """
        concat = lbann.Concatenation(original_nodes, updated_nodes, axis=1)
        attention_vector = self.nn_1(concat)
        attention_score = lbann.Softmax(attention_vector,
                                        softmax_mode="channel",
                                        name=self.name + "softmax")

        updated_nodes = self.nn_2(updated_nodes)
        updated_nodes = lbann.Multiply(attention_score, updated_nodes,
                                       name=self.name + "_output")

        return updated_nodes


class SGCNN(Module):
    """docstring for SGCNN"""
    def __init__(self,
                 num_nodes,
                 num_covalent_edges,
                 num_non_covalent_edges,
                 input_channels=19,
                 out_channels=1,
                 edge_feature_dim=1,
                 covalent_out_channels=20,
                 noncovalent_out_channels=30,
                 covalent_layers=1,
                 noncovalent_layers=1):
        super(SGCNN, self).__init__()
        self.num_nodes = num_nodes
        self.input_channels = input_channels

        cov_out = covalent_out_channels
        noncov_out = noncovalent_out_channels
        covalent_edge_nn = \
            [nn.ChannelwiseFullyConnectedModule(int(cov_out / 2), 
                                                           activation=lbann.Softsign),
                        nn.ChannelwiseFullyConnectedModule(cov_out * cov_out, 
                                                           activation=lbann.Softsign)
                        ]
        noncovalent_edge_nn = \
            [nn.ChannelwiseFullyConnectedModule(int(noncov_out / 2)),
                        lbann.Softsign,
                        nn.ChannelwiseFullyConnectedModule(noncov_out * noncov_out),
                        lbann.Softsign
                        ]

        self.covalent_propagation = Graph_Conv(input_feature_dim=input_channels,
                                               output_feature_dim=covalent_out_channels,
                                               edge_feature_dim=edge_feature_dim,
                                               num_layers=covalent_layers,
                                               num_nodes=num_nodes,
                                               num_edges=num_covalent_edges,
                                               edge_nn=covalent_edge_nn)

        self.non_covalent_propagation = Graph_Conv(input_feature_dim=covalent_out_channels,
                                                   output_feature_dim=noncovalent_out_channels,
                                                   edge_feature_dim=edge_feature_dim,
                                                   num_layers=noncovalent_layers,
                                                   num_nodes=num_nodes,
                                                   num_edges=num_non_covalent_edges,
                                                   edge_nn=noncovalent_edge_nn)

        self.cov_attention = Graph_Attention(covalent_out_channels,
                                             covalent_out_channels)
        self.noncov_attention = Graph_Attention(noncovalent_out_channels,
                                                noncovalent_out_channels)
        self.fully_connected_mlp = \
            Sequential([nn.FullyConnectedModule(int(noncov_out / 1.5)),
                        lbann.Relu,
                        nn.FullyConnectedModule(int(noncov_out / 2)),
                        lbann.Relu,
                        nn.FullyConnectedModule(out_channels)])
        self.gap = global_add_pool(num_nodes)

    def forward(self,
                node_features,
                edge_features,
                covalent_edge_sources,
                covalent_edge_targets,
                non_covalent_edge_sources,
                non_covalent_edge_targets,
                ligand_id_mask,
                fused=False):
        """Applies the SGCNN model to the PDB dataset
            Args: 
                node_features (Layer):
                edge_features (Layer): 
                covalent_edge_sources (Layer): 
                covalent_edge_targets (Layer): 
                non_covalent_edge_sources (Layer):
                non_covalent_edge_targets (Layer): 
                ligand_id_mask (Layer): 
                fused (bool) (default: False)
            returns: 
                ( Layer) 
        """
        node_features_cov = self.covalent_propagation(node_features,
                                                      edge_features,
                                                      covalent_edge_sources,
                                                      covalent_edge_targets)

        node_features = self.cov_attention(node_features_cov, node_features)

        node_features_noncov = self.non_covalent_propagation(node_features,
                                                             edge_features,
                                                             non_covalent_edge_sources,
                                                             non_covalent_edge_targets)

        node_features = self.noncov_attention(node_features_noncov, node_features)
      
        node_features_ligand_only = lbann.MatMul(ligand_id_mask, node_features)

        if(fused):
            return node_features_ligand_only
        else:
            x = self.fully_connected_mlp(node_features_ligand_only)
            return x

import lbann
from lbann.util import str_list


def slice_graph_data(input_layer,
                     num_nodes=4,
                     num_cov_edges=10,
                     num_non_cov_edges=20, 
                     node_features=19,
                     edge_features=1):
    """Utility function to slice the graph structured Protein data 
        Args: 
            input_layer (Layer): The input layer to 
            num_nodes (int): 
            num_cov_edges (int): 
            num_non_cov_edges (int): 
            node_features (int): 
            edge_features (int): 
        Returns: 
            dict: Returns a dictionary of LBANN Layer objects
    """
    
    slice_points = []
    slice_points.append(0)

    # Slice points for node features
    
    node_ft_end = num_nodes * node_features

    slice_points.append(node_ft_end)

    # Slice points for covalent adj matrix
    
    cov_adj_mat_sources = node_ft_end + num_cov_edges 

    cov_adj_mat_targets = cov_adj_mat_sources + num_cov_edges 
    
    slice_points.append(cov_adj_mat_sources)
    slice_points.append(cov_adj_mat_targets)
    
    # Slice points for noncovalent adj matrix
    
    noncov_adj_mat_sources = cov_adj_mat_targets + num_non_cov_edges

    noncov_adj_mat_target = noncov_adj_mat_sources + num_non_cov_edges
    
    slice_points.append(noncov_adj_mat_sources)
    slice_points.append(noncov_adj_mat_target)
    
    # Slice points for edge features

    edge_ft_end = noncov_adj_mat_target + num_non_cov_edges * edge_features

    slice_points.append(edge_ft_end)

    # Slice points for edge_adjacencies
    # This should be num_nodes * (num_nodes ** 2)
    prev_end = edge_ft_end
  
    # Slice points for ligand_only mat
    ligand_only_nodes = prev_end + num_nodes

    slice_points.append(ligand_only_nodes)

    # Slice for binding energy target
    target_end = ligand_only_nodes + 1
    
    slice_points.append(target_end)

    sliced_input = \
        lbann.Slice(input_layer, slice_points=str_list(slice_points))

    node_fts = \
        lbann.Identity(sliced_input, name="Node_fts_input")

    node_fts = \
        lbann.Reshape(node_fts, dims=str_list([num_nodes, node_features]), name="Node_fts_mat")

    cov_adj_sources = lbann.Identity(sliced_input, name="Covalent_Adj_sources_input")
    cov_adj_targets = lbann.Identity(sliced_input, name="Covalent_Adj_targets_input")

    cov_adj_sources = lbann.Reshape(cov_adj_sources, dims=str_list([num_cov_edges]), name="Covalent_Adj_sources")
    cov_adj_targets = lbann.Reshape(cov_adj_targets, dims=str_list([num_cov_edges]), name="Covalent_Adj_targets")

    noncov_adj_mat_sources = lbann.Reshape(lbann.Identity(sliced_input), dims=str_list([num_non_cov_edges]), name="Noncovalent_adj_sources")
    noncov_adj_mat_targets = lbann.Reshape(lbann.Identity(sliced_input), dims=str_list([num_non_cov_edges]), name="Noncovalent_adj_targets")

    edge_fts_inp = lbann.Identity(sliced_input)

    edge_fts = lbann.Reshape(edge_fts_inp, dims=str_list([num_non_cov_edges, 1, edge_features]), name="Edge_FTS")
    
    ligand_only_nodes = lbann.Reshape(lbann.Identity(sliced_input), dims=str_list([1, num_nodes]), name="Ligand_only_nodes")
    
    target = lbann.Reshape(lbann.Identity(sliced_input), dims="1", name="Target")

    sgcnn_data = {
        "node_features": node_fts,
        "edge_features": edge_fts,
        "covalent_sources": cov_adj_sources,
        "covalent_targets": cov_adj_targets,
        "non_covalent_sources": noncov_adj_mat_sources,
        "non_covalent_targets": noncov_adj_mat_targets,
        "ligand_only_nodes": ligand_only_nodes,
        "target": target 
    }

    return sgcnn_data


def slice_3D_data(data,
                  grid_size=48,
                  num_features=19):
    num_elements = (grid_size ** 3 * num_features)

    slice_points = str_list([0, num_elements, num_elements + 1])
    sliced_data = lbann.Slice(data, slice_points=slice_points)
    x = lbann.Identity(sliced_data, name="data_sample")
    y = lbann.Identity(sliced_data, name="target")
    x = lbann.Reshape(x,
                      dims="{0} {1} {1} {1}".format(grid_size, num_features))

    return x, y


def slice_FAST_data(data,
                    grid_size=48,
                    num_features=1,
                    num_nodes=50,
                    node_features=19,
                    edge_features=1):
    node_feat_mat = num_nodes * node_features
    edge_feat_mat = int((num_nodes * (num_nodes-1))/2) * edge_features
    edge_adj = num_nodes ** 3
    covalent_mat = num_nodes ** 2
    non_covalent_mat = num_nodes ** 2
    ligand_only_mat = num_nodes ** 2

    sample_size = (grid_size ** 3 * num_features) \
        + node_feat_mat + edge_feat_mat + edge_adj + covalent_mat \
        + non_covalent_mat + ligand_only_mat + 1
    num_elements = (grid_size ** 3 * num_features)

    slice_points = str_list([0, num_elements, sample_size])
    sliced_data = lbann.Slice(data, slice_points=slice_points)
    grid_data = lbann.Identity(sliced_data, name="grid_data_sample")
    graph_data = lbann.Identity(sliced_data, name="graph_data_sample")

    fast_data = {} 
    grid_data = \
        lbann.Reshape(grid_data,
                      dims="{0} {1} {1} {1}".format(grid_size, num_features))

    fast_data['grid'] = grid_data

    sgcnn_data = slice_graph_data(graph_data)

    fast_data.update(sgcnn_data,
                     num_nodes=num_nodes,
                     node_features=node_features,
                     edge_features=edge_features)

    return fast_data

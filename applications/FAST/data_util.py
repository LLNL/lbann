import lbann
from lbann.util import str_list
from lbann.modules import GraphVertexData

def slice_graph_data(input_layer,
                     num_nodes=4,
                     num_edges=10,
                     node_features=19,
                     edge_features=1):
    
    slice_points = []

    # Slice points for node features
    
    node_ft_end = num_nodes * node_features

    slice_points.append(node_ft_end)

    # Slice points for covalent adj matrix
    
    cov_adj_mat_sources = node_ft_end + num_edges 

    cov_adj_mat_targets = cov_adj_mat_sources + num_edges 
    
    slice_points.append(cov_adj_mat_sources)
    slice_points.append(cov_adj_mat_targets)
    
    # Slice points for noncovalent adj matrix
    
    noncov_adj_mat_sources = cov_adj_mat_targets + num_edges

    noncov_adj_mat_target = noncov_adj_mat_sources + num_edges
    
    slice_points.append(noncov_adj_mat_sources)
    slice_points.append(noncov_adj_mat_target)
    
    # Slice points for edge features

    edge_ft_end = noncov_adj_mat_end + num_edges * edge_features

    slice_points.append(edge_ft_end)

    
    # Slice points for edge_adjacencies
    # This should be num_nodes * (num_nodes ** 2)
    prev_end = edge_ft_end
    edge_adj = [(prev_end+(i+1)*(num_nodes**2)) for i in range(num_nodes)]
    prev_end = edge_adj[-1]
    # Slice points for ligand_only mat
    edge_adj_end = edge_adj[-1]
    ligand_only_end = edge_adj_end + (num_nodes ** 2)
    ligand_only = [ligand_only_end]
    # Slice for binding energy target
    target_end = ligand_only_end + 1
    target = [target_end]

    
    sliced_input = \
        lbann.Slice(input_layer, slice_points=str_list(slice_points))

    node_fts = \
        [lbann.Identity(sliced_input, name="Node_{}".format(i))
         for i in range(num_nodes)]

    cov_adj_mat = lbann.Identity(sliced_input, name="Covalent_Adj")

    noncov_adj_mat = lbann.Identity(sliced_input, name="NonCovalent_Adj")

    edge_fts = \
        [lbann.Identity(sliced_input, name="Edge_{}".format(i))
         for i in range(num_edges)]

    edge_adj = \
        [lbann.Identity(sliced_input, name="Adj_Mat_{}".format(i))
         for i in range(num_nodes)]

    ligand_ID = lbann.Identity(sliced_input, name="Ligand_only_nodes")

    target = lbann.Identity(sliced_input, name="Target")

    node_fts = \
        [lbann.Reshape(i, dims=str_list([1, node_features]))
         for i in node_fts]

    cov_adj_mat = \
        lbann.Reshape(cov_adj_mat, dims=str_list([num_nodes, num_nodes]))

    noncov_adj_mat = \
        lbann.Reshape(noncov_adj_mat, dims=str_list([num_nodes, num_nodes]))

    edge_features = \
        [lbann.Reshape(i, dims=str_list([1, edge_features])) for i in edge_fts]
    edge_adj = \
        [lbann.Reshape(i, dims=str_list([num_nodes, num_nodes]))
         for i in edge_adj]
    ligand_only = \
        lbann.Reshape(ligand_ID, dims=str_list([num_nodes, num_nodes]))
    target = lbann.Reshape(target, dims="1")

    node_fts = GraphVertexData(node_fts, node_features)
    return node_fts, cov_adj_mat, noncov_adj_mat, \
        edge_features, edge_adj, ligand_only, target


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

    grid_data = \
        lbann.Reshape(grid_data,
                      dims="{0} {1} {1} {1}".format(grid_size, num_features))
    node_ft, cov_adj, noncov_adj, edge_ft, edge_adj, ligand_only, target = \
        slice_graph_data(graph_data)

    return grid_data, node_ft, \
        cov_adj, noncov_adj, edge_ft, edge_adj, ligand_only, target

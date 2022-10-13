import urllib.request
import tarfile
import zipfile
import os.path
import numpy as np 

def download_url(url, save_path):
    """Helper function to download file from url and save it 
       on save_path
    """
    with urllib.request.urlopen(url) as dl_file:
        with open(save_path, 'wb') as out_file:
            out_file.write(dl_file.read()) 

def untar_file(data_dir, file_name):
    """Helper function to untar file
    """
    tar_name = os.path.join(data_dir, file_name)        
    with tarfile.open(tar_name) as tar:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar)
            tar.close()
def unzip_file(file_name, data_dir=None):
    """Helper function to unzip file
    """
    if (data_dir is None):
        data_dir = os.path.dirname(file_name)
                
    with zipfile.ZipFile(file_name, 'r') as zip_ref:
        zip_ref.extractall(data_dir)

def edge_list_to_dense(elist, num_vertices = 75):
    """ Generates an (num_vertices, num_vertices) adjacency 
        matrix given edge list, elist
    """

    adj_mat = np.zeros((num_vertices,num_vertices), dtype=np.float)
    num_edges = elist.shape[0]
    for edge in range(num_edges):
        source, sink = elist[edge,:]
        source = source.item()
        sink = sink.item()
        adj_mat[source][sink] = 1.0 
        adj_mat[sink][source] = 1.0
    return adj_mat

########################################################
#
# TU Dataset specific functions
#
########################################################

def extract_node_features(node_slices, node_labels, max_nodes, num_classes = None):
    node_label_list = [] 
    for i, ind in enumerate(node_slices[1:]):
        if num_classes:
            graph_x = np.eye(num_classes)[np.asarray([int(x) for x in node_labels[node_slices[i]:ind]],dtype=np.int)]
        else:
            graph_x = anp.asarray([int(x) for x in node_labels[node_slices[i]:ind]],dtype=np.int)
        if (len(graph_x) < max_nodes):
            pad = max_nodes - len(graph_x)
            graph_x = np.pad(graph_x, ((0,pad),(0,0)), 'constant')
            node_label_list.append(graph_x)
    return node_label_list 


def extract_adj_mat(node_slices, edge_list, max_nodes):
    adj_mat_list = []
    removed_graphs = []
    for i, max_node_id in enumerate(node_slices[1:]):
        min_node_id = node_slices[i]
        num_nodes = max_node_id - min_node_id
        if (num_nodes < max_nodes):
            edges = edge_list[(edge_list[:,1] > min_node_id) & (edge_list[:,1] < max_node_id)]
            edges = edges -1 - min_node_id 
            adj_mat = edge_list_to_dense(edges, max_nodes)
            adj_mat_list.append(adj_mat)
        else:
            removed_graphs.append(i)
            
    return adj_mat_list, removed_graphs
def extract_coo_list(node_slices, edge_list, max_nodes):
    source_indices_list_list = []
    target_indices_list_list = []
    removed_graphs = []

    for i, max_node_id in enumerate(node_slices[1:]):
        min_node_id = node_slices[i] 
        num_nodes = max_node_id - min_node_id
        if (num_nodes < max_nodes):
            edge = edge_list
            edges = edge_list[(edge_list[:,1] > min_node_id) & (edge_list[:,1] < max_node_id)]
            edges = edges -1 - min_node_id
            source_indices_list_list.append(edges[:,0])
            target_indices_list_list.append(edges[:,1])
        else:
            removed_graphs.append(i)
    return source_indices_list_list, target_indices_list_list, removed_graphs


def extract_targets(graph_labels, num_classes, removed_graphs):
    graph_labels = np.array([int(x) for x in graph_labels])
    labels = np.eye(num_classes)[graph_labels-1]
    graph_labels =  np.delete(labels, removed_graphs, axis=0)
    return graph_labels

def dataset_node_slices(graph_indicator_list, num_graphs):
    node_slices = []
    
    prev = 0
    for i in range(num_graphs+1):
        node_slices.append(prev+graph_indicator_list.count(str(i)))
        prev = prev + graph_indicator_list.count(str(i))
    return node_slices

def TUDataset_Parser(data_dir,
                     dataset_name,
                     num_classes,
                     max_nodes = 100,
                     graph_format="sparse"):

    def data_extract(description):
        with open(os.path.join(data_dir, f"{dataset_name}_{description}.txt"),'r') as _fd:
            _data = _fd.read().rstrip().split('\n')
        return _data

    adj_list = data_extract("A")
    graph_labels = data_extract("graph_labels")
    graph_ind = data_extract("graph_indicator")
    node_attr = data_extract("node_attributes")
    node_labels = data_extract("node_labels")

    NUM_GRAPHS =  len(graph_labels)
    NUM_NODES = len(node_attr)
    NUM_EDGES = len(adj_list)

    edge_list = [] 
    for edge in adj_list:
        edge = np.array([int(x) for x in edge.split(',')])
        edge_list.append(edge)

    edge_list = np.array(edge_list)

    node_slices = dataset_node_slices(graph_ind, NUM_GRAPHS)
    num_features = 3
    node_features = extract_node_features(node_slices, node_labels,max_nodes, num_features)
    node_features = np.array(node_features)

    if graph_format == 'dense':
        adj_mat, removed_graphs = extract_adj_mat(node_slices, edge_list, max_nodes)
        targets = extract_targets(graph_labels, num_classes, removed_graphs)
        return (node_features, adj_mat, targets)
    else:
        source_node_indices, target_node_inidices, removed_graphs = \
            extract_coo_list(node_slices, edge_list, max_nodes)
        targets = extract_targets(graph_labels, num_classes, removed_graphs)
        return (node_features, source_node_indices, target_node_inidices, targets )

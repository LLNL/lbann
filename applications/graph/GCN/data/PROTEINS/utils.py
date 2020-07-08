import urllib.request
import tarfile
import zipfile
import os.path
import numpy as np 

def download_url(url, save_path):
    with urllib.request.urlopen(url) as dl_file:
        with open(save_path, 'wb') as out_file:
            out_file.write(dl_file.read()) 

def untar_file(data_dir, file_name):
    tar_name = os.path.join(data_dir, file_name)        
    with tarfile.open(tar_name) as tar:
            tar.extractall()
            tar.close()
def unzip_file(file_name, data_dir=None):
    if (data_dir is None):
        data_dir = os.path.dirname(file_name)
                
    with zipfile.ZipFile(file_name, 'r') as zip_ref:
        zip_ref.extractall(data_dir)

def edge_list_to_dense(elist, num_vertices = 75):
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

def TUDataset_Parser(data_dir, dataset_name, num_classes):
        
    adj_file = open(os.path.join(data_dir, dataset_name + '_A.txt'), 'r')
    graph_labels_file = open(os.path.join( data_dir, dataset_name + '_graph_labels.txt'), 'r')
    graph_ind_file = open(os.path.join( data_dir, dataset_name + '_graph_indicator.txt'), 'r')    
    node_attr_file = open(os.path.join( data_dir, dataset_name + '_node_attributes.txt'), 'r')
    node_labels_file = open(os.path.join( data_dir, dataset_name + '_node_labels.txt'), 'r')

    graph_labels = graph_labels_file.read().rstrip().split('\n')
    graph_ind = graph_ind_file.read().rstrip().split('\n')
    node_attr = node_attr_file.read().rstrip().split('\n')
    adj_list = adj_file.read().rstrip().split('\n')
    node_labels = node_labels_file.read().rstrip().split('\n')

    NUM_GRAPHS =  len(graph_labels)
    NUM_NODES = len(node_attr)
    NUM_EDGES = len(adj_list)

    adj_file.close()
    graph_labels_file.close()
    graph_ind_file.close()
    node_attr_file.close()
    node_labels_file.close()
    edge_list = [] 
    for edge in adj_list:
        edge = np.array([int(x) for x in edge.split(',')])
        edge_list.append(edge)

    edge_list = np.array(edge_list)

    node_slices = dataset_node_slices(graph_ind, NUM_GRAPHS)
    
    max_nodes = 100
    adj_mat, removed_graphs = extract_adj_mat(node_slices, edge_list, max_nodes)
    num_features = 3
    node_features = extract_node_features(node_slices, node_labels,max_nodes, num_features)
    node_features = np.array(node_features)
    targets = extract_targets(graph_labels, num_classes, removed_graphs)

    return node_features, adj_mat, targets

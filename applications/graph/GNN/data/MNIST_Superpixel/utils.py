import torch 
import urllib.request
import tarfile 
import os
import os.path
import numpy as np
import lbann 
 
data_dir = os.path.dirname(os.path.realpath(__file__))

def download_data():
    url = "http://ls7-www.cs.uni-dortmund.de/cvpr_geometric_dl/mnist_superpixels.tar.gz"
    training_name = "training.pt"
    test_name = "test.pt"

    files = [training_name, test_name]

    for f in files:
        data_file = os.path.join(data_dir, f)

        if not os.path.isfile(data_file): #File not in directory 
            tar_name = os.path.join(data_dir, "mnist_superpixel.tar.gz")

            if not os.path.isfile(tar_name):
                urllib.request.urlretrieve(url, filename=tar_name)
                extract_data()
            else:
                extract_data()

def extract_data():
     tar_name = os.path.join(data_dir, "mnist_superpixel.tar.gz") 
     print(tar_name)
     with tarfile.open(tar_name) as tar:
        tar.extractall()
        tar.close()
def edge_list_to_dense(elist):
    adj_mat = np.zeros((75,75), dtype=np.float)

    ## elist should be of shape (2, num_edges) 

    num_edges = elist.size(1)

    for edge in range(num_edges):
        source, sink = elist[:,edge]
        source = source.item()
        sink = sink.item()
        adj_mat[source][sink] = 1.0
        adj_mat[sink][source] = 1.0
    
    return adj_mat

def process_training_data(): # Process Training File
    train_file_path = os.path.join(data_dir, 'training.pt')
    #test_file_path = os.path.join(data_dir, 'test.pt')
    
    node_features, edge_index, edge_slices, positions, y = torch.load(train_file_path)
    
    assert y.size(0) == node_features.size(0)
    assert y.size(0) == positions.size(0)
    assert y.size(0) == 60000 ## 

    num_data = 60000
    num_vertices = 75
        # Nodes features should be (60000, 75)
        
    node_features = np.float32(node_features)
        
        # Position should be (60000, 75, 2)

    positions = np.float32(positions)

        # Convert edge_index to edge matrix representation with shape (60000, 75, 75)
        
    adj_matrices = np.zeros( (num_data, num_vertices, num_vertices), dtype=np.float)

    #assert (self.num_data + 1) == edge_slices.size(0), "Expected: {}, Got{} ".format(60001, edge_slices.size(0))
        
    for slice_index in range(num_data):
        print("{}/{} completed \r".format(slice_index+1, num_data), end='',flush=True)
        start_index = edge_slices[slice_index]
        end_index = edge_slices[slice_index + 1]

        graph_num = slice_index
        elist = edge_index[:, start_index: end_index ]

        adj_matrices[graph_num] = edge_list_to_dense(elist)


        # Convert y to target with one hot encoding and shape (60000, 10)

    targets = np.zeros ( (num_data, 10), dtype=np.float)

    for i, target in enumerate(y):
        print("{}/{} completed".format(i+1, len(y)), end='') 
        targets[i][target] = 1

    np.save('node_features.npy',node_features)
    np.save('positions.npy',positions)
    np.save('adj_matrices.npy', adj_matrices)
    np.save('targets.npy',targets)

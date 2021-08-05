import numpy as np 
import os 
import os.path 
import sys 
import utils
import numpy as np 
import sys 


data_dir = os.path.dirname(os.path.realpath(__file__))

def get_data():
    if not os.path.isfile(os.path.join(data_dir, "PROTEINS.zip")):
        #Needs Download
        url = 'https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/PROTEINS.zip'
        save_path = os.path.join(data_dir, 'PROTEINS.zip')
        utils.download_url(url, save_path)
    utils.unzip_file(os.path.join(data_dir, "PROTEINS.zip"))
        


class PROTEINS_Sparse_Dataset():
    files = ['node_features_sparse.npy', 
             'source_indices_sparse.npy',
             'target_indices_sparse.npy',
             'targets_sparse.npy']
    
    def __init__(self):
        # Check is data is downloaded and processed
        # Load if data exists 
        # Else Download and process data
        files = PROTEINS_Sparse_Dataset.files  
        for npy_file in files:
            if not os.path.isfile(os.path.join(data_dir,"PROTEINS/"+npy_file)):
                get_data()
                self.generate_dataset()
                break

        self.node_features = np.load(os.path.join(data_dir, "PROTEINS/"+files[0]))
        self.source_indices = np.load(os.path.join(data_dir,"PROTEINS/"+files[1]))
        self.target_indices = np.load(os.path.join(data_dir,"PROTEINS/"+files[2]))
        self.targets = np.load(os.path.join(data_dir, "PROTEINS/"+files[3]))
        self.max_edges = max([x.shape[0] for x in self.source_indices])
        self.pad_edge_lists()

    def generate_dataset(self):
        files = PROTEINS_Sparse_Dataset.files
        save_dir = os.path.join(data_dir, 'PROTEINS')        
        data = utils.TUDataset_Parser(save_dir, 'PROTEINS', 2, graph_format="sparse")
        
        for file_name, _graph_data_i in zip(files, data):
            np.save(os.path.join(save_dir, file_name), _graph_data_i)

    def pad_edge_lists(self):
        for i, (sources, targets) in enumerate(zip(self.source_indices, self.target_indices)):
            padding_size = self.max_edges - sources.shape[0]
            padding = np.full((padding_size), -1)
            self.source_indices[i] = np.concatenate((sources, padding), axis=0)
            self.target_indices[i] = np.concatenate((targets, padding), axis=0)

    def __len__(self):
        return len(self.node_features)

    def __getitem__(self, index):
        x = np.float32(self.node_features[index].flatten())
        s = np.float32(self.source_indices[index].flatten())
        t = np.float32(self.target_indices[index].flatten())
        y = np.float32(self.targets[index].flatten())
        return np.concatenate((x,s,t,y), axis=0)             



class PROTEINS_Dense_Dataset():
    files = ['node_features_dense.npy', 'adj_mats_dense.npy', 'targets_dense.npy']
    
    def __init__(self):
        # Check is data is downloaded and processed
        # Load if data exists 
        # Else download and process data
        files = PROTEINS_Dense_Dataset.files  
        for npy_file in files:
            if not os.path.isfile(os.path.join(data_dir,"PROTEINS/"+npy_file)):
                get_data()
                self.generate_dataset()
                break
        self.node_features = np.load(os.path.join(data_dir, "PROTEINS/"+files[0]))
        self.adjs = np.load(os.path.join(data_dir,"PROTEINS/"+files[1]))
        self.targets = np.load(os.path.join(data_dir, "PROTEINS/"+files[2]))

    def generate_dataset(self):
        files = PROTEINS_Dense_Dataset.files
        save_dir = os.path.join(data_dir, 'PROTEINS')        
        data = utils.TUDataset_Parser(save_dir, 'PROTEINS', 2, graph_format="dense")
        
        for file_name, _graph_data_i in zip(files, data):
            np.save(os.path.join(save_dir, file_name), _graph_data_i)
    
    def __len__(self):
        return len(self.node_features)
    
    def __getitem__(self, index):
        x = np.float32(self.node_features[index].flatten())
        y = np.float32(self.targets[index].flatten())
        adj = np.float32(self.adjs[index].flatten())

        return np.concatenate((x,adj,y), axis=0)

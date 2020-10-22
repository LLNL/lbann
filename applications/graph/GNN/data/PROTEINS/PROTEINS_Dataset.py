import numpy as np 
import os 
import os.path 
import sys 
import utils
import numpy as np 
import sys 

files = ['node_features.npy', 'adj_mats.npy', 'targets.npy']

data_dir = os.path.dirname(os.path.realpath(__file__))

class PROTEINS_Dataset:
    def __init__(self):
        # Check is data is downloaded and processed
        # Load if data exists 
        # Else Download and process data  
        for npy_file in files:
            if not os.path.isfile(os.path.join(data_dir,"PROTEINS/"+npy_file)):
                self.process_data()

        self.node_features = np.load(os.path.join(data_dir, "PROTEINS/"+files[0]))
        self.adjs = np.load(os.path.join(data_dir,"PROTEINS/"+files[1]))
        self.targets = np.load(os.path.join(data_dir, "PROTEINS/"+files[2]))
       
    def generate_dataset(self):
        global data_dir
        print(data_dir)
        data_dir = os.path.join(data_dir, 'PROTEINS')        
        node_features, adj_mat, targets = utils.TUDataset_Parser(data_dir, 'PROTEINS', 2)
        np.save(os.path.join(data_dir, files[0]), node_features)
        np.save(os.path.join(data_dir, files[1]), adj_mat)
        np.save(os.path.join(data_dir, files[2]), targets)

    def process_data(self):
        if not os.path.isfile(os.path.join(data_dir, "PROTEINS.zip")):
            #Needs Download
            url = 'https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/PROTEINS.zip'
            save_path = os.path.join(data_dir, 'PROTEINS.zip')
            utils.download_url(url, save_path)
        utils.unzip_file(os.path.join(data_dir, "PROTEINS.zip"))
        
        self.generate_dataset()

    def __len__(self):
        
        return len(self.node_features)
    def __getitem__(self, index):
        
        x = np.float32(self.node_features[index].flatten())
        y = np.float32(self.targets[index].flatten())
        adj = np.float32(self.adjs[index].flatten())

        return np.concatenate((x,adj,y), axis=0)

training_data = PROTEINS_Dataset()

def get_train(index):
    return training_data[index]

def num_train_samples():
    return len(training_data)

def sample_dims():
    adjacency_matrix_size = 100 * 100 
    node_feature_size = 100 * 3 
    target_size = 2
    return (adjacency_matrix_size + node_feature_size + target_size, )

if __name__== '__main__':
    print(len(training_data))
    print(training_data.node_features[0].shape)
    print(training_data.adjs[0].shape)
    print(training_data.targets[0].shape)
    print(type(training_data[0][0]))
    print(sys.getsizeof(training_data[0][0]))

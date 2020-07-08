import numpy as np 
import os 
import os.path 
import sys 
import utils
import numpy as np 

files = ['node_features.npy', 'adj_mats.npy', 'targets.npy']

data_dir = os.path.dirname(os.path.realpath(__file__))

class PROTEINS_Dataset:
    def __init__(self):
        # Check is data is downloaded and processed
        # Load if data exists 
        # Else Download and process data  
        for npy_file in files:
            if not os.path.isfile(os.path.join(data_dir,npy_file)):
                self.process_data()

        self.node_features = np.load(os.path.join(data_dir, files[0]))
        self.adjs = np.load(os.path.join(data_dir,files[1]))
        self.targets = np.load(os.path.join(data_dir, files[2]))
       
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

    
    def __getitem__(self, index):
        
        x = self.node_features.flatten()
        y = self.targets.flatten()
        adj = self.adjs.flatten()

        return np.concatenate((x,adj,y), axis=0)
if __name__== '__main__':
    dataset = PROTEINS_Dataset()

    

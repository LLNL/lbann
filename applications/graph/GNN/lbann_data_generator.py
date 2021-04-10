import numpy as np
import torch
import ogb
import matplotlib.pyplot as plt
from ogb.graphproppred.mol_encoder import AtomEncoder,BondEncoder
from tqdm import tqdm
from torch_geometric.data import Data


from ogb.lsc import PCQM4MDataset
from ogb.utils import smiles2graph


# convert each SMILES string into a molecular graph object by calling smiles2graph
# This takes a while (a few hours) for the first run
dataset = PCQM4MDataset(root = ROOT, smiles2graph = smiles2graph)

_data = torch.load("pcqm4m_kddcup2021/processed/data_processed")
data_split_indices = torch.load('pcqm4m_kddcup2021/split_dict.pt')

training = data_split_indices['train']
validation = data_split_indices['valid']


validation_data = []
for index in tqdm(validation):
    graph = _data['graphs'][index]
    num_nodes = graph['num_nodes'] 
    homolumogap = _data['labels'][index]
    data = Data()
    

    data.__num_nodes__ = int(graph['num_nodes'])
    data.edge_index = torch.from_numpy(graph['edge_index']).to(torch.int64)
    data.edge_attr = torch.from_numpy(graph['edge_feat']).to(torch.int64)
    data.x = torch.from_numpy(graph['node_feat']).to(torch.int64)
    data.y = torch.Tensor([homolumogap])
    
    validation_data.append(data)
with open('LSC_PCQM4M/valid.bin','wb') as f:
    pickle.dump(validation_data, f)
    



for name, _set in [("training", training), ("validation",validation)]:
    
    filename = 'LBANN_Data_'+name+'.bin'
    count = 1
    fp = np.memmap(filename, dtype='float32', mode='w+', shape=(len(_set),1101))
    for index in tqdm(training):
        graph = _data['graphs'][index]
        num_nodes = graph['num_nodes'] 

        nodes = -1 * np.ones((51,9), dtype=np.float32)
        nodes[: num_nodes, :] = np.float32(graph['node_feat'])

        edges = -1 * np.ones((118,3), dtype=np.float32)

        num_edges = graph['edge_feat'].shape[0] 
        edges[:num_edges,:] = graph['edge_feat']


        sources = -1 * np.ones(118, dtype=np.float32)
        targets = -1 * np.ones(118, dtype=np.float32)
        sources[:num_edges] = graph['edge_index'][0]
        targets[:num_edges] = graph['edge_index'][1]

        mask = np.zeros(51, dtype=np.float32)
        mask[:num_nodes] = 1
        label = np.array([training_labels[index]], dtype=np.float32)
        fp[index, :] = np.concatenate([(nodes.T).flatten(), (edges.T).flatten(), sources, targets, mask, label])

        if count % 10000 == 0:
            fp.flush()
        count += 1
    fp.flush()
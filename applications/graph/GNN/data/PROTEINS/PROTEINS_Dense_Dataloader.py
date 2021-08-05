from PROTEINS_Dataset import PROTEINS_Dense_Dataset

protein_data = PROTEINS_Dense_Dataset()

def get_train(index):
  return protein_data[index]

def num_train_samples():
  return len(protein_data)


def sample_data_dims():
    adjacency_matrix_size = 100 * 100 
    node_feature_size = 100 * 3 
    target_size = 2
    return (adjacency_matrix_size + node_feature_size + target_size, )

if __name__== '__main__':
    print(len(protein_data))
    print(protein_data.node_features[0].shape)
    print(protein_data.adjs[0].shape)
    print(protein_data.targets[0].shape)
    print(type(protein_data[0][0]))

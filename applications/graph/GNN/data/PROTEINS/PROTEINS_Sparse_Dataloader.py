from PROTEINS_Dataset import PROTEINS_Sparse_Dataset

protein_data = PROTEINS_Sparse_Dataset()

def get_train(index):
	return protein_data[index]

def num_train_samples():
	return len(protein_data)

def sample_data_dims():
  node_feature_size = 100 * 3
  target_size = 2 
  max_source_nodes = protein_data.max_edges
  max_target_nodes = max_source_nodes
  return (node_feature_size+target_size+max_source_nodes+max_target_nodes,)

if __name__ == '__main__':
	print("Dataset info: ")
	print("Total Number of samples: ", num_train_samples())
	print("Sample size: ", sample_data_dims())
	print("Max number of edges: ",protein_data.max_edges )

	print(get_train(0).shape)
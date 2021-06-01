import numpy as np 

dataset = np.array(np.memmap("/p/vast1/zaman2/LBANN_Data.bin",
            dtype='float32',
            mode='r',
            shape=(3045360,1101)))

def get_sample_func(index):
  return dataset[index]


def num_samples_func():
  return 3045360


def sample_dims_func():
  return (1101, )


if __name__ == '__main__':
  split_indices = []

  start_index = 0

  split_indices.append(start_index)

  node_features = [51 for i in range(1, 10) ]

  split_indices.extend(node_features)

  edge_features = [118 for i in range(1,4)]

  split_indices.extend(edge_features)
  misc = [118, 118, 1]
  split_indices.extend(misc)


  for i in range(1, len(split_indices)):
    split_indices[i] = split_indices[i] + split_indices[i-1]
  print(split_indices)

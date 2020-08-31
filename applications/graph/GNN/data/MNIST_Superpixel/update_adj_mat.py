import numpy as np

adj_mats = np.load('adj_matrices.npy')

num_data = adj_mats.shape[0]
for adj in range(num_data):
    print(adj, " / ", num_data)
    deg_inv_sqrt = (adj_mats[adj].sum(axis=-1).clip(min=1)**(-0.5)).reshape(len(adj_mats[adj]),1)
    adj_mats[adj] =deg_inv_sqrt*adj_mats[adj]*deg_inv_sqrt
np.save('adj_matrices.npy', adj_mats)

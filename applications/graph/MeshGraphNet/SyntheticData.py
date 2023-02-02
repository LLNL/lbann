import numpy as np 


NUM_SAMPLES = 10000
NUM_NODES = 100
NUM_EDGES = 1000
NODE_FEATS = 5
EDGE_FEATS = 3
OUT_FEATS = 3

NODE_FEATURE_SIZE = NUM_NODES * NODE_FEATS
EDGE_FEATURE_SIZE = NUM_EDGES * EDGE_FEATS
OUT_FEATURE_SIZE =  NUM_EDGES * OUT_FEATS

def get_sample_func(index):
  random_features = np.random.random(NODE_FEATURE_SIZE+OUT_FEATURE_SIZE).astype(np.float32)
  source_indices = np.random.randint(-1, NUM_NODES, size=NUM_EDGES).astype(np.float32)
  target_indices = np.random.randint(-1, NUM_NODES, size=NUM_EDGES).astype(np.float32)
  out_features = np.random.random(EDGE_FEATURE_SIZE).astype(np.float32)

  return np.concatenate([random_features, source_indices, target_indices, out_features])

def num_samples_func():
  return NUM_SAMPLES

def sample_dims_func():

  size = NODE_FEATURE_SIZE + EDGE_FEATURE_SIZE + OUT_FEATURE_SIZE + 2 * NUM_EDGES
  return (size, )

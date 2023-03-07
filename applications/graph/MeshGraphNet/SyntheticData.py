import numpy as np 
import configparser


DATA_CONFIG = configparser.ConfigParser()
DATA_CONFIG.read("data_config.ini")
NUM_NODES = 100 # int(DATA_CONFIG['DEFAULT']['NUM_NODES'])
NUM_EDGES = 10000 # int(DATA_CONFIG['DEFAULT']['NUM_EDGES'])
NODE_FEATS = 5 # int(DATA_CONFIG['DEFAULT']['NODE_FEATURES'])
EDGE_FEATS = 3 # int(DATA_CONFIG['DEFAULT']['EDGE_FEATURES'])
OUT_FEATS = 3 # int(DATA_CONFIG['DEFAULT']['OUT_FEATURES'])
NUM_SAMPLES = 100



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


if __name__ == '__main__':
    print(NUM_NODES)

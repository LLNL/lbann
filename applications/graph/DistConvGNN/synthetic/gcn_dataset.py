import numpy as np
import configparser


config = configparser.ConfigParser()
config.read('gcn_data_config.ini')
NUM_ROWS = config['DEFAULT']['NumVertices']
NUM_COLS = config['DEFAULT']['NumFeats']
INDICES_VEC = config['DEFAULT']['NumEdges'] 


NUM_SAMPLES = 15

_data = np.arange(NUM_SAMPLES * NUM_ROWS * NUM_COLS).reshape((NUM_SAMPLES, NUM_ROWS, NUM_COLS))
_synth_target = np.randint(10, size=NUM_SAMPLES)
_targets = np.zeros((NUM_SAMPLES, 10))
_targets[np.arange(NUM_SAMPLES), _synth_target] = 1

_indices = np.tile(np.array([0, 1, 2, 3]), 2 * NUM_SAMPLES * INDICES_VEC // 4).reshape(NUM_SAMPLES, 2 * INDICES_VEC)


def num_train_samples():
  return NUM_SAMPLES


def sample_dims():
  return (NUM_ROWS * NUM_COLS + INDICES_VEC * 2 + 10, )


def get_sample(index):
  return np.concatenate([_data[index].flatten(), _indices[index], _targets[index].flatten()]) .astype(np.float32)

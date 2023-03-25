import numpy as np
import configparser


config = configparser.ConfigParser()
config.read('data_config.ini')
NUM_ROWS = config['DEFAULT']['NumRows']
NUM_COLS = config['DEFAULT']['NumCols']
OUT_ROWS = config['DEFAULT']['OutRow'] 
MODE = config['DEFAULT']['Mode']


NUM_SAMPLES = 15
INDICES_VEC = OUT_ROWS

if MODE == 'SCATTER':
  INDICES_VEC = NUM_ROWS

_data = np.arange(NUM_SAMPLES * NUM_ROWS * NUM_COLS).reshape((NUM_SAMPLES, NUM_ROWS, NUM_COLS))
_indices = np.tile(np.array([0, 1, 2, 3]), NUM_SAMPLES * INDICES_VEC // 4).reshape(NUM_SAMPLES, INDICES_VEC)

def num_train_samples():
  return NUM_SAMPLES


def sample_dims():
  return (NUM_ROWS * NUM_COLS + INDICES_VEC, )


def get_sample(index):
  return np.concatenate([_data[index].flatten(), _indices[index]]) .astype(np.float32)


if __name__ == '__main__':
  print("Testing a simple Scatter implementation ")
  samples = get_sample(0) 
  mat_size = NUM_ROWS * NUM_COLS
  values = samples[:mat_size].reshape(NUM_ROWS, NUM_COLS)
  indices = samples[mat_size:].astype(np.int32)
  output = np.zeros((OUT_ROWS, NUM_COLS))

  for i in range(NUM_ROWS):
    output[indices[i]] += values[i]

  print("input tensor: \n", values)
  print("indices: \t", indices)
  print("output tensor: \n", output)
 

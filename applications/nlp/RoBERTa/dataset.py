import numpy as np

data = np.random.randint(50265, size=(1024,513))
data[:,0] = data[:,0] % 10

def get_sample(i):
    vals = data[i]
    return vals.flatten().astype(np.float32)

def num_samples():
    return 1024

def sample_dims():
    return (513,)


import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

np.seterr(divide='ignore', invalid='ignore')

samples = np.load('norm_1_train_val.npy', mmap_mode='r')[:,0,...]

def inv_transform(x):
    return 4 * (1 + x) / (1 - x) - 1

dim = samples.shape[-1]
ki = np.fft.fftfreq(dim)
k = np.sqrt(ki[:,None,None]**2 + ki[None,:,None]**2 + ki[None,None,:]**2)
bins = np.linspace(0, 1, 65)
counts = np.histogram(k.ravel(), bins)[0]

def compute_pk(i):
    np.seterr(divide='ignore', invalid='ignore')
    x = inv_transform(samples[i,...])
    fk = np.square(np.abs(np.fft.fftn(x)))
    pk = np.histogram(k.ravel(), bins, weights=fk.ravel())[0] / counts
    return pk

pks = np.array(Parallel(n_jobs=44)(delayed(compute_pk)(i) for i in tqdm(range(len(samples)))))

pk_rel = pks / pks.mean(axis=0, keepdims=True)

np.save('target_pk.npy', np.nan_to_num(pks.mean(axis=0)))
k_weights = np.nan_to_num(1 / pk_rel.var(axis=0))
k_weights /= k_weights.sum()
np.save('k_weights.npy', k_weights)

import numpy as np
from glob import glob
import os
from PIL import Image
from tqdm import tqdm

base_dir = os.path.dirname(os.path.realpath(__file__))

# Get the most recent training run directory.
exp_dirs = glob(os.path.join(base_dir, '20*'))
exp_dirs.sort(key=os.path.getctime)

# Get all saved generator samples.
files = glob(os.path.join(exp_dirs[-1], 'dump_outs', 'trainer0', 'model0', '*.npy'))
files.sort(key=os.path.getctime)

# Combine samples into one large array with 16 samples for each training step.
all_samps = []
for f in tqdm(files):
    samps = np.load(f)
    samps = np.concatenate([samps[i] for i in range(16)], axis=-1)
    all_samps.append(samps)
all_samps = np.concatenate(all_samps, axis=1).transpose(1, 2, 0)

img = Image.fromarray(np.uint8(all_samps * 255))
img.save(os.path.join(base_dir, 'samps.jpg'))
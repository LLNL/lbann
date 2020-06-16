import functools
import operator
import os.path
import random
import sys
import cv2
import numpy as np
from .extract_patches import extract_patches
from .patterns import patterns_2patch, patterns_3patch, patterns_4patch, patterns_5patch
from .chroma_blur import chroma_blur

# Data paths
label_file = '/p/lscratchh/brainusr/ILSVRC2012/labels/train.txt'
data_dir = '/p/lscratchh/brainusr/ILSVRC2012/original/train'

# Read label files
samples = []
with open(label_file) as f:
    for line in f:
        line = line.split(' ')
        samples.append((line[0], int(line[1])))

# Get sample function
def get_sample_2patch(index):
    return get_sample(index, 2)
def get_sample_3patch(index):
    return get_sample(index, 3)
def get_sample_4patch(index):
    return get_sample(index, 4)
def get_sample_5patch(index):
    return get_sample(index, 5)
def get_sample(index, num_patches):
    """Generate data sample.

    Extract patches and apply preprocessing tricks.
    """

    # Read image from file
    file_name, _ = samples[index]
    file_name = os.path.join(data_dir, file_name)
    img = cv2.imdecode(np.fromfile(file_name, dtype=np.uint8),
                       cv2.IMREAD_COLOR)

    # Crop to get square image
    size = min(img.shape[0], img.shape[1])
    y = (img.shape[0] - size) // 2
    x = (img.shape[1] - size) // 2
    img = img[y:y+size, x:x+size, :]

    # Extract patches
    patterns = None
    if num_patches == 2:
        patterns = patterns_2patch
    if num_patches == 3:
        patterns = patterns_3patch
    if num_patches == 4:
        patterns = patterns_4patch
    if num_patches == 5:
        patterns = patterns_5patch
    patches, label = extract_patches(img, patterns)

    # Randomly rotate patches
    rotate_type = random.randint(0, 3)
    for i, patch in enumerate(patches):
        patch = np.rot90(patch, rotate_type, axes=(0,1))
        patches[i] = patch
    label = label + rotate_type * len(patterns)

    # Convert patch to float32
    for i, patch in enumerate(patches):
        if patch.dtype == np.uint8:
            patches[i] = patch.astype(np.float32) / 255

    # Chroma blur
    for i, patch in enumerate(patches):
        patches[i] = chroma_blur(patch)

    # Transform to CHW format and normalize
    for i, patch in enumerate(patches):
        patch = np.transpose(patch, axes=(2, 0, 1))
        means = np.array([0.406, 0.456, 0.485]).reshape((3,1,1))
        stdevs = np.array([0.225, 0.224, 0.229]).reshape((3,1,1))
        patch -= means
        patch /= stdevs
        patches[i] = patch

    # Random aperture
    for i, patch in enumerate(patches):
        if i == 0:
            continue
        size = random.randint(64, 96)
        y = random.randint(0, 96-size)
        x = random.randint(0, 96-size)
        new_patch = np.zeros((3, 96, 96), dtype=np.float32)
        new_patch[:, y:y+size, x:x+size] = patch[:, y:y+size, x:x+size]
        patches[i] = new_patch

    # Construct one-hot label vector
    label_vec = np.zeros(num_labels(num_patches), dtype=np.float32)
    label_vec[label] = 1

    # Return flattened data tensors
    flat_data = []
    for patch in patches:
        flat_data.append(patch.reshape(-1))
    flat_data.append(label_vec)
    return np.concatenate(flat_data)

# Get sample dims functions
patch_dims = (3, 96, 96)
def num_labels(num_patches):
    num_patterns = 0
    if num_patches == 2:
        num_patterns = len(patterns_2patch)
    if num_patches == 3:
        num_patterns = len(patterns_3patch)
    if num_patches == 4:
        num_patterns = len(patterns_4patch)
    if num_patches == 5:
        num_patterns = len(patterns_5patch)
    return 4 * num_patterns
def sample_dims(num_patches):
    patch_size = functools.reduce(operator.mul, patch_dims)
    return (num_patches*patch_size + num_labels(num_patches),)
def sample_dims_2patch():
    return sample_dims(2)
def sample_dims_3patch():
    return sample_dims(3)
def sample_dims_4patch():
    return sample_dims(4)
def sample_dims_5patch():
    return sample_dims(5)

# Get num samples function
def num_samples():
    return len(samples)

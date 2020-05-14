import enum
import math
import random
import cv2
import numpy as np

# ----------------------------------------------
# Patch type specification
# ----------------------------------------------
# Note: Sizes and positions are in pixels.

class PatchType(enum.Enum):
    _3X3 = 1
    _2X2 = 2
    OVERLAP = 3

# 3x3-type patches
_3x3_patch_pos = ((0.0,     0.0), (0.0,     137/384), (0.0,     274/384),
                  (137/384, 0.0), (137/384, 137/384), (137/384, 274/384),
                  (274/384, 0.0), (274/384, 137/384), (274/384, 274/384))
_3x3_patch_size = 110/384

# 2x2-type patches
_2x2_patch_pos = ((0,       0), (0,       146/256),
                  (146/256, 0), (146/256, 146/256))
_2x2_patch_size = 110/256

# Overlap-type patches
overlap_patch_pos = ((0,      0), (0,      86/196),
                     (86/196, 0), (86/196, 86/196))
overlap_patch_size = 110/196

# ----------------------------------------------
# Patch extraction
# ----------------------------------------------

def extract_patch(img, patch_type, index, zoom, jitter):
    """Extract a patch from image and resize.

    Args:
        img (ndarry): Image in HWC format.
        patch_type (PatchType): Desired patch type.
        index (int): Patch index.
        zoom (float): Zoom factor.
        jitter ((float, float)): Jitter positions, normalized in
            [0,1).

    Returns:
        ndarray: Patch in HWC format.

    """

    # Get patch position
    if patch_type == PatchType._3X3:
        posy = _3x3_patch_pos[index][0]
        posx = _3x3_patch_pos[index][1]
        patch_size = _3x3_patch_size
    if patch_type == PatchType._2X2:
        posy = _2x2_patch_pos[index][0]
        posx = _2x2_patch_pos[index][1]
        patch_size = _2x2_patch_size
    if patch_type == PatchType.OVERLAP:
        posy = overlap_patch_pos[index][0]
        posx = overlap_patch_pos[index][1]
        patch_size = overlap_patch_size

    # Apply zoom and jitter to patch position
    posy += (1-1/zoom) * patch_size * jitter[0]
    posx += (1-1/zoom) * patch_size * jitter[1]
    patch_size /= zoom

    # Identify patch pixels
    img_size = img.shape[0]
    y0 = math.floor(posy * img_size)
    y1 = math.ceil((posy + patch_size) * img_size)
    x0 = math.floor(posx * img_size)
    x1 = math.ceil((posx + patch_size) * img_size)
    y0 = max(0, min(img_size-1, y0))
    y1 = max(1, min(img_size, y1))
    x0 = max(0, min(img_size-1, x0))
    x1 = max(1, min(img_size, x1))

    # Extract patch from image
    interp_methods = (cv2.INTER_LINEAR, cv2.INTER_AREA,
                      cv2.INTER_CUBIC, cv2.INTER_LANCZOS4)
    patch = cv2.resize(img[y0:y1, x0:x1, :],
                       (96, 96),
                       interpolation=random.choice(interp_methods))

    # Randomly apply horizontal flip
    if random.choice([True, False]):
        patch = np.fliplr(patch)

    return patch

def extract_patches(img, patterns):
    """Extract patches from image.

    Args:
        img (ndarry): Image in HWC format.
        patterns (list of (list of (PatchType, int))): Patch patterns.
            See patterns.py.

    Returns:
        list of ndarray: Patches in HWC format.
        int: Patch pattern label.

    """

    label = random.randint(0, len(patterns)-1)
    zoom = random.uniform(1, 128/96)
    jitter = (random.random(), random.random())
    patches = [extract_patch(img, p[0], p[1], zoom, jitter)
               for p in patterns[label]]
    random.shuffle(patches)
    return patches, label

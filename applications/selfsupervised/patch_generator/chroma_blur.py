import numpy as np
import scipy.ndimage.filters
import cv2

def chroma_blur(img):
    """Blur chroma channels to hide chromatic aberration.

    Convert to CIE Lab format and apply box filter to a and b
    channels.

    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    img[:,:,1] = scipy.ndimage.filters.uniform_filter(img[:,:,1], 13)
    img[:,:,2] = scipy.ndimage.filters.uniform_filter(img[:,:,2], 13)
    img = cv2.cvtColor(img, cv2.COLOR_Lab2BGR)
    return img

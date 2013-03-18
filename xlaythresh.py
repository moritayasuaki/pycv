# -*- coding: utf-8 -*-
from skimage import io, exposure, filter, segmentation, morphology

from pylab import imshow, show
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.ndimage.filters as scifilters
import numpy as np
import scipy.ndimage.measurements as meas

def sigmoid(x):
  return 1/(1+np.exp(-x))

def sigab(img, a, b):
  return sigmoid(a*(img-b))

img = io.imread("./senegal/1457-1875h/x-lay/1S.jpg",as_grey=True)
img = sigab(img.astype(np.float32)/256, 20, 0.23)
img1 = scifilters.median_filter(img,5)
img1 = (img1 < 0.22)
img1 = morphology.remove_small_objects(img1).astype(np.float32)
imshow(img,cmap=cm.Greys_r)
show()
imshow(img1,cmap=cm.Greys_r)
show()

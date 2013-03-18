# -*- coding: utf-8 -*-
from skimage import io, exposure, filter, morphology

from pylab import imshow, show
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.ndimage.filters as scifilters
import numpy as np
import scipy.ndimage.measurements as meas
from scipy import ndimage

def sigmoid(x):
  return 1/(1+np.exp(-x))

def sigab(img, a, b):
  return sigmoid(a*(img-b))

img = io.imread("pics/x-lay/100826_ArgentinaNewStarter-SW_500H_n=32/1S.jpg",as_grey=True)

# stretch histogram
img = (img-np.min(img))/(np.max(img)-np.min(img))

# get histgram shape
n, bins = np.histogram(img, bins=np.arange(0,255)/256.0)

# smoothing
n = scifilters.gaussian_filter(n.astype(np.float32),3)

# histogram test plot
plt.plot(bins[:len(n)],n)
show()

# get local maxima and minima
maxima = (n[:-2] <= n[1:-1]) * (n[1:-1] >= n[2:])
minima = (n[:-2] >= n[1:-1]) * (n[1:-1] <= n[2:])
argmaxima = []
argminima = []
for i in xrange(len(n[1:-1])):
  if maxima[i]:
    argmaxima.append(bins[i+1])
  if minima[i]:
    argminima.append(bins[i+1])

# print extrema
print argmaxima
print argminima

# distribute marker points
markers = np.zeros_like(img)
markers[img < (argminima[0]+argmaxima[0])/2] = 2
markers[img > argmaxima[1]] = 1

# sobel edge detector
em = filter.sobel(img)

# watershd segmentation
seg = morphology.watershed(em,markers)

# eliminate small cluster
seg = ndimage.binary_fill_holes(seg - 1)

# label cluster
labeled_img, _ = ndimage.label(seg)

# original image
imshow(img,cmap=cm.Greys_r)
show()
# labeled image
imshow(labeled_img)
show()

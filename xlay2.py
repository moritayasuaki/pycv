#!/usr/bin/python
import numpy as np
import scipy.ndimage as ndimg
import scipy.signal as ssig
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage import data, io, filter, measure, morphology, segmentation, color, exposure, draw, feature
import sys

debug = True

def debshow(img):
  io.imshow(img)
  io.show()

if debug:
  fname = "./data/S12057-1_s.jpg"
else:
  fname = sys.argv[1]

image = io.imread(fname,as_grey=True)
edges = filter.sobel(image)

hist,bincs = exposure.histogram(image,nbins=100)

hsmoothed = ndimg.filters.gaussian_filter1d(hist,sigma=1)

local_maxi = feature.peak_local_max(hsmoothed,exclude_border=False,threshold_rel=0.0,threshold_abs=0.0)

if debug:
  local_max = feature.peak_local_max(hsmoothed,indices=False,exclude_border=False,threshold_rel=0.0,threshold_abs=0.0)
  plt.plot(bincs,hsmoothed)
  plt.plot(bincs,local_max*1024)
  plt.show()

markers = np.zeros_like(image)
for i in local_maxi:
  mask = image < (bincs[i] + 0.05)
  mask = mask * image > (bincs[i] - 0.05)
  markers = markers + mask*(i+1)

io.imshow(markers)
io.show()

seg = morphology.watershed(edges,markers)
seg = ndimg.binary_fill_holes(seg-1)

labeled, _ = ndimg.label(seg)

plt.imshow(labeled)
plt.show()


markers = filter.rank.gradient(image,morphology.disk(5)) < 10
markers = ndimage.label(markers)[0]
gradient = filter.rank.gradient(image.morphology.disk(2))

labels = morphology.watershed(gradient,markers)

plt.imshow(labels)
plt.show()

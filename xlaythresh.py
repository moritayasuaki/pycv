#!/usr/local/bin/python
# -*- coding: utf-8 -*-
from skimage import io, exposure, filter, morphology, feature

from pylab import imshow, show
import sys
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.ndimage.filters as scifilters
import scipy.signal as sig
import numpy as np
import scipy.ndimage.measurements as meas
from scipy import ndimage
from scipy import misc

debug = False
argc = len(sys.argv)
if argc != 2:
  print 'please input arguments'
  quit()

fname = sys.argv[1]

# def sigmoid(x):
  # return 1/(1+np.exp(-x))

# def sigab(img, a, b):
  # return sigmoid(a*(img-b))

img = io.imread(fname,as_grey=True)

# img = filter.median_filter(img,radius=2)

# downsample
img = misc.imresize(img,(200,400)).astype(np.float32)

l = np.max(img) - np.min(img)

# stretch histogram
img = (img-np.min(img))/(np.max(img)-np.min(img))


# get histgram shape
n, bins = np.histogram(img, bins=np.arange(0,99)/100.0)

# histogram test plot
if debug:
  plt.plot(bins[:len(n)],n)
  show()

# smoothing
n = scifilters.gaussian_filter(n.astype(np.float32),1.5)


if debug:
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

print argmaxima
print argminima

# distribute marker points
markers = np.zeros_like(img)
markers[img < (argminima[0]+argmaxima[0])/2] = 2
markers[img > argmaxima[1]] = 1

# sobel edge detector
em = filter.sobel(img)

# watershed segmentation
seg = morphology.watershed(em,markers)
seg = ndimage.binary_fill_holes(seg - 1)

# label cluster
labeled_img, labels = ndimage.label(seg)


# maximum cluster
sizes = np.bincount(labeled_img.ravel())
sizes[0] = 0
ssizes = (np.argsort(sizes))[::-1]

# electrodes

electrode0 = labeled_img == ssizes[0]
electrode1 = labeled_img == ssizes[1]


# electron destribution

s_electrode0 = electrode0 - morphology.binary_erosion(electrode0,morphology.disk(1))
y0,x0 = np.ma.nonzero(s_electrode0)
s_set0 = zip(y0,x0)
s_electrode1 = electrode1 - morphology.binary_erosion(electrode1,morphology.disk(1))
y1,x1 = np.ma.nonzero(s_electrode1)
s_set1 = zip(y1,x1)

def calc_electric_field(y,x):
  ev = np.zeros(img.shape,dtype=np.complex)
  for (ye,xe) in s_set0:
    dv = (y-ye)+(x-xe)*1j
    d = np.abs(dv)
    ev += dv/(d**2)
  for (ye,xe) in s_set1:
    dv = -((y-ye)+(x-xe)*1j)
    d = np.abs(dv)
    ev += dv/(d**2)
  return ev

# electric field
efield = np.fromfunction(calc_electric_field, img.shape, dtype=np.float32)
efield[np.nonzero(labeled_img)] = 0.0
epower = np.abs(efield)

import numpy.random as rand

print morphology.is_local_maximum

# original image

# labeled image
if debug:
  imshow(labeled_img)
  show()

# cleaned image
if debug:
  imshow(s_electrode0)
  show()

  imshow(s_electrode1)
  show()

# electric power field
plt.figure(1)
plt.subplot(211)
imshow(img,cmap=cm.Greys_r)
plt.subplot(212)
imshow(epower)
plt.colorbar()
plt.savefig(fname + ".png")
show()

#!/usr/bin/env python
# coding:utf-8

import os
import sys
import numpy as np
from skimage import io
from skimage import exposure
from skimage import transform
from skimage.filter import rank
from sklearn import preprocessing
from sklearn import decomposition
from sklearn import manifold
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


n_points = 1000
n_neighbors = 4
n_components = 3

def main(argv):
  filename = argv[1]
  img = io.imread(filename, as_grey=True)
  lpyra = tuple(transform.pyramid_laplacian(img))
  l = lpyra[0]
  l = exposure.equalize_hist(l)
  y, x = np.indices((l.shape[0],l.shape[1]))
  vect = np.array(zip(y.reshape(y.size),x.reshape(x.size),l.reshape(l.size)))
  io.imshow(l)
  io.show()

def plot(original):
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.scatter(original[:,0],original[:,1],zs = original[:,2], c=np.arange(len(original)), cmap=plt.cm.Spectral)
  # ax.set_xlim(-2.0,2.0)
  # ax.set_ylim(-2.0,2.0)
  # ax.set_zlim(-2.0,2.0)
  plt.show()

main(sys.argv)

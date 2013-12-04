#!/usr/bin/env python
# coding:utf-8

import os
import sys
import numpy as np
from skimage import io
from sklearn import preprocessing
from sklearn import decomposition
from sklearn import manifold
import matplotlib.pyplot as plt

def main(argv):
  filename = argv[1]
  img = io.imread(filename,as_grey=True)
  randy = np.random.randint(img.shape[0],size=2000)
  randx = np.random.randint(img.shape[1],size=2000)
  vect = np.array([[randy[i],randx[i],img[randy[i],randx[i]]] for i in np.arange(2000)])
  vect = preprocessing.scale(vect)
  vect_pca = decomposition.PCA(n_components=3).fit_transform(vect)
  x, y = vect_pca.T
  plt.scatter(x,y,s=2)
  plt.show()

main(sys.argv)

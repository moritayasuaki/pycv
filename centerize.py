# -*- coding: utf-8 -*-
# centerize.py

import numpy as np
from numpy.fft import fft2, ifft2, fftshift
from skimage import color
from skimage import io
from skimage import draw
from scipy import ndimage as ndi
from scipy import signal
from pylab import imshow, show

testim = io.imread("./pics/senegal/3003-2585h/appearance/1R.jpg")
testarr = np.array([[1,2,3],[4,5,6],[7,8,9]])

def hamming2d(shape):
  wndh = np.array([signal.hamming(shape[0])])
  wndw = np.array([signal.hamming(shape[1])])
  return np.dot(wndh.T, wndw)

def rot180(gray):
  """
  >>> rot180(testarr)
  array([[9, 8, 7],
         [6, 5, 4],
         [3, 2, 1]])
  """
  return gray[::-1,::-1]

def psymPoint(im):
  """
  >>> y,x = psymPoint(testim)
  >>> rr,cc = draw.circle(y,x,5)
  >>> testim[rr,cc] = 0
  >>> imshow(testim)
  >>> show()
  """
  l,a,b = color.rgb2lab(im.astype(np.float32)/256).transpose(2,0,1)
  window = hamming2d(l.shape)
  img = b * window
  rimg = rot180(img)
  fimg = fft2(img)
  frimg = fft2(rimg)
  fcorr = fimg * frimg.conj()
  fcorr /= np.abs(fcorr)
  dy, dx = ndi.measurements.maximum_position( np.abs(ifft2(fcorr)) )
  if (dy > l.shape[0]/2):
    dy = l.shape[0]/2 - dy
  if (dx > l.shape[1]/2):
    dx = l.shape[1]/2 - dx
  dy /= 2
  dx /= 2
  return l.shape[0]/2+dy, l.shape[1]/2+dx

if __name__ == "__main__":
  import doctest
  doctest.testmod()


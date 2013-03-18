# -*- coding: utf-8 -*-
# centerize.py

import numpy as np
import math
from numpy.fft import fft2, ifft2, fftshift
from skimage import color
from skimage import io
from skimage import draw
from scipy import ndimage as ndi
from scipy.ndimage import interpolation as ndii
from scipy import signal
from pylab import imshow, show

testim = io.imread("./pics/senegal/3003-2585h/appearance/1R.jpg")
testim2 = io.imread("./pics/senegal/3003-2585h/appearance/2R.jpg")
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


def pcorr(im1,im2):
  fim1 = fft2(im1)
  fim2 = fft2(im2)
  fcorr = fim1 * fim2.conj()
  fcorr /= np.abs(fcorr)
  return np.abs(ifft2(fcorr))


def psymPoint(im):
  """
  >>> y,x = psymPoint(testim)
  >>> rr,cc = draw.circle(y,x,10)
  >>> testim[rr,cc] = 0
  >>> imshow(testim)
  >>> show()
  """
  l,a,b = color.rgb2lab(im.astype(np.float32)/256).transpose(2,0,1)
  window = hamming2d(l.shape)
  l = ndi.filters.gaussian_filter(l,8,mode='wrap')
  img = l * window # + ndi.filters.gaussian_filter(l,16,mode='wrap') * (1 - window)
  show()
  rimg = rot180(img)
  dy, dx = ndi.measurements.maximum_position(pcorr(img,rimg))
  if (dy > l.shape[0]/2):
    dy -= l.shape[0]
  if (dx > l.shape[1]/2):
    dx -= l.shape[1]
  dy /= 2
  dx /= 2
  return l.shape[0]/2+dy, l.shape[1]/2+dx


def logpolar(image, center=None, angles=None, radii=None):
  """Return log-polar transformed image and log base."""
  shape = image.shape
  if center is None:
    center = shape[0]/2, shape[1]/2
  if angles is None:
    angles = shape[0]
  if radii is None:
    radii = shape[1]
  theta = np.empty((angles, radii), dtype=np.float64)
  theta.T[:] = -np.linspace(0, np.pi, angles, endpoint=False)
  #d = radii
  d = np.hypot(shape[0]-center[0], shape[1]-center[1])
  log_base = 10.0 ** (math.log10(d) / (radii))
  radius = np.empty_like(theta)
  radius[:] = np.power(log_base, np.arange(radii, dtype=np.float64)) - 1.0
  x = radius * np.sin(theta) + center[0]
  y = radius * np.cos(theta) + center[1]
  output = np.empty_like(x)
  ndii.map_coordinates(image, [x, y], output=output)
  return output, log_base


def scaleRotMatch(im1, im2, center1=None, center2=None):
  """Return rotScale"""
  shape1 = im1.shape
  shape2 = im2.shape
  if center1 is None:
    center1 = shape1[0]/2, shape1[1]/2
  if center2 is None:
    center2 = shape2[0]/2, shape2[1]/2
  lpim1,lb1 = logpolar(im1, center1)
  lpim2,lb2 = logpolar(im2, center2)
  imshow(lpim1)
  show()
  imshow(lpim2)
  show()


l,a,b = color.rgb2lab(testim.astype(np.float32)/256).transpose(2,0,1)
scaleRotMatch(l,a)

if __name__ == "__main__":
  import doctest
  doctest.testmod()


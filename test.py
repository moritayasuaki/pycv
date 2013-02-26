from cv2 import *
import cv2
from numpy import *
import numpy
import sys
import scipy
from scipy import misc, special
from scipy.ndimage import filters

def sigmoid(x):
  return 1/(1+exp(-x))

def nsigmoid(x):
  r = x.max()-x.min()
  m = median(x)
  return sigmoid((x-m)/r*3)

def normalize(x):
  return 0.5*(1+special.erf((x-x.mean())/x.std()/2))


fnames = sys.argv[1:]
imgs=[]
for fname in fnames:
  bgrimg = cv2.imread(fname).astype(float32)/256.0
  (h,w,d) = bgrimg.shape
  bgrimg = cv2.resize(GaussianBlur(bgrimg,(5,5),1.4),(w/4,h/4))
  imgs.append(bgrimg)



from cv2 import *
import cv2
from numpy import *
import numpy
import sys
import scipy
from scipy import misc, special
from scipy.ndimage import filters
import numpy.linalg
def sigmoid(x):
  return 1/(1+exp(-x))

def nsigmoid(x):
  return 1/(1+exp(-(x-x.mean())/x.std()))


class PCA(object):
  def __init__(self, dim):
    self.dim = dim

  def calc(self, src):
    shape = src.shape
    mu = (sum(numpy.asarray(src,dtype=numpy.float32)) /float(len(src)))
    src_m = (src - mu).T
    if (shape[0] < shape[1]):
      if (shape[0] < self.dim): self.dim = shape[0]
      
      print>>sys.stderr,"convariance matrix..."
      n = numpy.dot(src_m.T, src_m)/float(shape[0])
      print>>sys.stderr,"done"

      print>>sys.stderr,"eigen value decomposition"
      l,v = numpy.linalg.eig(n)
      idx = l.argsort()
      l = l[idx][::-1]
      v = v[:,idx][:,::-1]
      print>>sys.stderr,"done"

      vm=numpy.dot(src_m, v)
      for i in range(len(l)):
        if l[i]<=0:
          v = v[:,:i]
          l = l[:i]
          if (self.dim < i): 
            self.dim = i
          break
        vm[:,i]=vm[:,i]/numpy.sqrt(shape[0]*l[i])
    else:
      if shape[1] < self.dim:
        self.dim = shape[1]
      cov = numpy.dot(src_m, src_m.T)/float(shape[0])
      l, vm = numpy.linalg.eig(cov)
      idx = l.argsort()
      l = l[idx][::-1]
      vm = vm[:,idx][:,::-1]

    self.eigenvectors = vm[:,:self.dim]
    self.eigenvalues = l[:self.dim]
    self.mean = mu
  def project(self,vec):
    return numpy.dot(self.eigenvectors.T,(vec-self.mean).T).T

  def backproject(self,vec):
    return (numpy.dot(self.eigenvalues,vec.T)).T+self.mean

fnames = sys.argv[1:]
imgs=[]
for fname in fnames:
  imgs.append(cv2.imread(fname).astype(numpy.float32))

data=[]
for img in imgs:
  data.append(numpy.reshape(img,256*256*3))

data = numpy.asarray(data)
print data.shape
pca = PCA(40)
pca.calc(data)
print pca.eigenvectors.shape
print pca.eigenvalues.shape
p1 = pca.project(data)
print p1.shape
v1 = pca.backproject(p1)
print v1.shape
# d,b = v1.shape
# for i in range(d):
  # img = v1[i,:,:,:]
  # img = numpy.reshape(img,(256,256,3))
  # cv2.imshow("test",img.astype(uint8))
  # cv2.waitKey(0)

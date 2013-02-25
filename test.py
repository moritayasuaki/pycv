from cv2 import *
import cv2
from numpy import *
import numpy
import sys

def sigmoid(x):
  return 1/(1+exp(-x))

def sigmoid_dash(x):
  return 2/(0.5*exp(x)+1+0.5*exp(-x))

def nsigmoid(x):
  return 1/(1+exp(-(x-x.mean())/x.std()))


fname = sys.argv[1]
rgbimg = imread(fname).astype(float32)/256.0


(h,w,d) = rgbimg.shape


ycrcbimg = cvtColor(rgbimg,COLOR_RGB2YCR_CB)

yimg = ycrcbimg[:,:,0]
crimg = ycrcbimg[:,:,1]
cbimg = ycrcbimg[:,:,2]

gyimg = GaussianBlur(yimg,(7,7),0)
gcrimg = GaussianBlur(crimg,(101,101),20)
gcbimg = GaussianBlur(cbimg,(101,101),20)

# gyimg = boxFilter(yimg,-1,(9,9))
# gcrimg = boxFilter(crimg,-1,(51,51))
# gcbimg = boxFilter(cbimg,-1,(51,51))

fimg = cvtColor(array([gyimg,gcrimg,gcbimg]).swapaxes(0,1).swapaxes(1,2),COLOR_YCR_CB2RGB)


halffimg = cv2.resize(fimg,(w/4,h/4))


hsvimg = cvtColor(halffimg,COLOR_RGB2HSV)

himg = hsvimg[:,:,0]
simg = hsvimg[:,:,1]
vimg = hsvimg[:,:,2]
svimg = hsvimg[:,:,1:3]

imshow(fname,cv2.resize(rgbimg,(w/4,h/4)))
moveWindow(fname,0,0)
imshow("filtered",halffimg)
moveWindow("filtered",w/4,0)
imshow("simg",nsigmoid(simg))
moveWindow("simg",0,h/4)
imshow("vimg",nsigmoid(vimg))
moveWindow("vimg",w/4,h/4)
imshow("himg",himg/180)
moveWindow("himg",w/4,h/2)
waitKey(0)

import cv2
import sys
import numpy as np
from scipy import special


def preprocess(img):
  print "prepocessed.."
  dimg = downSample(img)
  inpaints, mask = reflectionMask(dimg)
  clbpos = calibration(inpaints)
  dimg[:,clbpos]=np.array([0,0,0])
  cv2.imshow("dimg",dimg)
  cv2.imshow("mask",mask)
  cv2.imshow("inpaints",inpaints)
  cv2.waitKey(0)

def objSample(img):
  electrode(ccimg)
  whitemud(ccimg)
  blackmud(ccimg)
  grain(ccimg)
  mercury(ccimg)

def downSample(img):
  h,w,d = np.shape(img)
  return cv2.resize(img,(w/4,h/4))

def reflectionMask(img):
  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  ret, mask = cv2.threshold(gray, 0.75, 1, cv2.THRESH_BINARY)
  uimg = (256*img).astype(np.uint8)
  img = cv2.inpaint(uimg, mask.astype(np.uint8), 10, cv2.INPAINT_TELEA)
  img.astype(np.float32)/256
  return (img,mask)


def calibration(img):
  h,w,d = np.shape(img)
  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  s = w/5
  mirror = np.fliplr(gray)[:,s:w-s]
  match = cv2.matchTemplate(gray,mirror,method=cv2.TM_CCOEFF_NORMED)
  d = np.argmax(match)
  centerx = (w+d-s)/2
  centery = 
  return center

pname, fname = sys.argv
print "load " + fname + " ..."
img = cv2.imread(fname).astype(np.float32)/256.0
preprocess(img)

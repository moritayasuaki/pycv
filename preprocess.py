import cv2
import sys
import numpy as np
from scipy import special
from scipy import misc
from os import path
def preprocess(img):
  print "prepocessed.."
  dimg = downSample(img)
  inpaints, mask = reflectionMask(dimg)
  (cx,cy,d) = calibration(inpaints)
  dimg2 = inpaints[cy-d:cy+d,cx-d:cx+d]
  dimg2 = cv2.resize(dimg2,(256,256))
  # cv2.imshow("mask",mask)
  # cv2.imshow("inpaints",inpaints)
  return dimg2

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
  lab = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
  l = lab[:,:,0]
  a = lab[:,:,1]
  b = lab[:,:,2]
  ul = (l*2.56).astype(np.uint8)
  ule = cv2.equalizeHist(ul)
  ret, mask = cv2.threshold(ule, 245, 1, cv2.THRESH_BINARY)
  iul = cv2.inpaint(ul, mask.astype(np.uint8), 10, cv2.INPAINT_TELEA)
  ifl = (iul.astype(np.float32))/2.56
  lab = np.array([ifl,a,b]).swapaxes(0,1).swapaxes(1,2)
  img = cv2.cvtColor(lab,cv2.COLOR_LAB2BGR)
  return (img,mask)


def calibration(img):
  h,w,d = np.shape(img)
  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  s = w/5
  mirror = np.fliplr(gray)[:,s:w-s]
  match = cv2.matchTemplate(gray,mirror,method=cv2.TM_CCOEFF_NORMED)
  d = np.argmax(match)
  centerx = (w+d-s)/2
  gray = cv2.equalizeHist((gray[h/4:h*3/4,centerx-w*2/5:centerx+w*2/5]*256).astype(np.uint8))
  m = cv2.moments(gray)
  centery = m['m01']/m['m00'] + h/4
  vary = m['mu20']
  varx = m['mu02']
  sd = np.sqrt(varx)
  print (centerx,centery,sd*0.0012)
  return (centerx,centery,sd*0.0012)

pname, outputdir, fname = sys.argv
bname = path.basename(fname)
print "load " + fname + " ..."
img = cv2.imread(fname).astype(np.float32)/256.0
cv2.imwrite(outputdir + "/" + bname, preprocess(img)*256)

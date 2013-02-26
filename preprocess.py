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
  dimg2 = inpaints[cy-d/2:cy+d,cx-d:cx+d]
  dimg2 = cv2.resize(dimg2,(256,192))
  # cv2.imshow("inpaints",dimg2)
  # cv2.waitKey(0)
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
  # lab = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
  # l = lab[:,:,0]
  # a = lab[:,:,1]
  # b = lab[:,:,2]
  # ul = (l*2.56).astype(np.uint8)
  # ua = ((100+a)*1.28).astype(np.uint8)
  # ub = ((100+b)*1.28).astype(np.uint8)
  # ule = cv2.equalizeHist(ul)
  # ret, mask = cv2.threshold(ule, 242, 255, cv2.THRESH_BINARY)
  # mask = mask.astype(np.uint8)
  # iul = cv2.inpaint(ul, mask, 20, cv2.INPAINT_TELEA)
  # iua = cv2.inpaint(ua, mask, 20, cv2.INPAINT_TELEA)
  # iub = cv2.inpaint(ub, mask, 20, cv2.INPAINT_TELEA)

  # ifl = (iul.astype(np.float32))/2.56
  # ifa = (iua.astype(np.float32)-128)/1.28
  # ifb = (iub.astype(np.float32)-128)/1.28

  # lab = np.array([ifl,ifa,ifb]).swapaxes(0,1).swapaxes(1,2)
  b = img[:,:,0]
  g = img[:,:,1]
  r = img[:,:,2]
  ret, mask = cv2.threshold(2*b+g+r,2.8,255, cv2.THRESH_BINARY)
  mask = mask.astype(np.uint8)
  mimg = cv2.inpaint((img*256).astype(np.uint8),mask, 10, cv2.INPAINT_TELEA)
  mimg = mimg.astype(np.float32)/256
  return (mimg,mask)


def normalizeHist(img):
  eq = cv2.equalizeHist((img*256).astype(np.uint8)).astype(np.float32)
  return 0.5*(1+special.erf((eq-eq.mean())/eq.std()))

def calibration(img):
  h,w,d = np.shape(img)
  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  s = w/5
  mirror = np.fliplr(gray)[:,s:w-s]
  match = cv2.matchTemplate(gray,mirror,method=cv2.TM_CCOEFF_NORMED)
  d = np.argmax(match)
  centerx = (w+d-s)/2
  ret, mask = cv2.threshold(gray,gray.mean()-5*gray.var(), 1, cv2.THRESH_BINARY)
  # cv2.imshow("equa",mask)
  # cv2.waitKey(0)
  m = cv2.moments(mask)
  centery = m['m01']/m['m00']
  vary = np.sqrt(m['nu02'])
  varx = m['nu20']
  sd = np.sqrt(vary)
  print (centerx,centery,0.8*h*sd)
  # cv2.waitKey(0)
  return (centerx,centery,0.8*h*sd)

pname, outputdir, fname = sys.argv
bname = path.basename(fname)
print "load " + fname + " ..."
img = cv2.imread(fname).astype(np.float32)/256.0
cv2.imwrite(outputdir + "/" + bname, preprocess(img)*256)

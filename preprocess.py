import cv2 as cv
import sys
import numpy as np
prog,fname = sys.argv
print "load " + fname + " ..."
img = cv.imread(fname)
print "loaded."
print type(img)
print "show " + fname

h,w,d=img.shape

# Noise reduction
img = cv.medianBlur(img,3)
img = cv.bilateralFilter(img,0,32,2)
# Down sizing
img = cv.resize(img,(w/4,h/4))

# float
img2=img.astype(np.float32)/256.0
# toHLS
hlsimg=cv.cvtColor(img2,cv.COLOR_BGR2HSV)


himg=hlsimg[:,:,0]
simg=hlsimg[:,:,1]
limg=hlsimg[:,:,2]


# limg(x,y) = 1.0 and simg(x,y) = 0 then mask
# limg(x,y) = 0.95 or simg(x,y) > 0.95 then mask

limg = (limg*256).astype(np.uint8)
simg = (simg*256).astype(np.uint8)
himg = himg.astype(np.uint8)

limg2 = cv.equalizeHist(limg)
ret,limg2 = cv.threshold(limg,240,255,cv.THRESH_BINARY)
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
mask = cv.dilate(limg2,kernel)


limg = cv.inpaint(limg,mask,2,cv.INPAINT_NS)
simg = cv.inpaint(simg,mask,2,cv.INPAINT_NS)
simg = cv.bilateralFilter(simg, 0,50,8)
#himg = cv.inpaint(himg,mask,2,cv.INPAINT_NS)
#himg = cv.bilateralFilter(himg, 0,50,8)
#himg = cv.GaussianBlur(himg, (7,7), 2)


# oimg = np.ones((h/4,w/4))
# simg = oimg - simg

cv.imshow("simg", simg)
cv.imshow("limg", limg)


hlsimg=np.array([himg,simg,limg])
hlsimg=hlsimg.swapaxes(0,1)
hlsimg=hlsimg.swapaxes(1,2)

cv.imshow("hls",cv.cvtColor(hlsimg,cv.COLOR_HSV2BGR))

limg3 = cv.equalizeHist(limg)
limg3 = limg.astype(np.float64)/256
simg3 = simg
simg3 = cv.equalizeHist(simg)
# simg3 = cv.GaussianBlur(simg3, (7,7), 2)
simg3 = simg3.astype(np.float64)/256
simg3 = np.ones((h/4,w/4)) - simg3*1.2
simg3 = np.power(simg3,3)
whiteimg = simg3*limg3
cv.imshow("hls3",limg3)
cv.imshow("hls2",simg3)


limg3mean = limg3.mean()
blackimg = np.exp(-(np.abs(limg3-limg3mean)-0.2))-np.exp(-(np.abs(limg3-limg3mean)-0.2)/3)
print blackimg

def dog(i,sigma):
  s2 = sigma/1.235
  s1 = s2*1.6
  return 2.15*(cv.GaussianBlur(i,(11,11),s1)-cv.GaussianBlur(i,(11,11),s2))

grainimg = dog(limg3,0.3).astype(np.float32)+dog(limg3,0.4).astype(np.float32)+dog(limg3,0.5).astype(np.float32)+dog(limg3,0.6).astype(np.float32)
grainimg = cv.resize(grainimg,(w/4,h/4))
img = cv.resize(img,(w/4,h/4))
# middle result
cv.imshow("down", img)
cv.imshow("whiteimg",whiteimg)
cv.moveWindow("whiteimg",w/4,0)
cv.imshow("grainimg",grainimg)
cv.moveWindow("grainimg",0,h/4)
cv.imshow("blackimg",blackimg*2)
cv.moveWindow("blackimg",w/4,h/4)

cimg = cv.cvtColor(hlsimg,cv.COLOR_HSV2BGR)
cv.waitKey(0)
#cv.imshow("img", cv.resize(cimg,(w/2,h/2)))
print "finished."

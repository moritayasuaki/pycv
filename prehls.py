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
img = cv.resize(img,(w/5,h/5))

cv.imshow("bi", img)
img=img.astype(np.float32)/256.0
hlsimg=cv.cvtColor(img,cv.COLOR_BGR2HLS)

himg=np.zeros((h/5,w/5))
limg=hlsimg[:,:,1]
simg=hlsimg[:,:,2]

simg = cv.GaussianBlur(simg,(5,5),2)
cv.imshow("simg", simg)

(ret,lmask)=cv.threshold(limg,0.93,1,cv.THRESH_BINARY)
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
lmask=cv.dilate(lmask,kernel)
cv.imshow("lmask", lmask)
cv.imshow("limg", limg)

himg=himg.astype(np.uint8)
simg=(simg*256).astype(np.uint8)
limg=(limg*256).astype(np.uint8)
lmask=lmask.astype(np.uint8)

hsvimg=np.array([himg,limg,simg])
hsvimg=hsvimg.swapaxes(0,1)
hsvimg=hsvimg.swapaxes(1,2)

cimg = cv.cvtColor(hsvimg,cv.COLOR_HLS2BGR)
cimg = cv.inpaint(cimg,lmask,2,cv.INPAINT_TELEA)

cv.imshow("img", cv.resize(cimg,(w/2,h/2)))
print "finished."
input()

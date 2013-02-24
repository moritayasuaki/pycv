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


himg=np.zeros((h/4,w/4))
simg=hlsimg[:,:,1]
limg=hlsimg[:,:,2]


# limg(x,y) = 1.0 and simg(x,y) = 0 then mask
# limg(x,y) = 0.95 or simg(x,y) > 0.95 then mask

limg = (limg*256).astype(np.uint8)
simg = (simg*256).astype(np.uint8)


limg2 = cv.equalizeHist(limg)
ret,limg2 = cv.threshold(limg2,240,255,cv.THRESH_BINARY)
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
mask = cv.dilate(limg2,kernel)

limg = cv.inpaint(limg,mask,2,cv.INPAINT_NS)
simg = cv.inpaint(simg,mask,2,cv.INPAINT_NS)
simg = cv.bilateralFilter(simg, 0,50,8)
simg = cv.GaussianBlur(simg, (7,7), 2)

himg=himg.astype(np.uint8)
hlsimg=np.array([himg,simg,limg])
hlsimg=hlsimg.swapaxes(0,1)
hlsimg=hlsimg.swapaxes(1,2)

limg3 = limg.astype(np.float64)/256
simg3 = simg.astype(np.float64)/256

lmean = limg3.mean()
smean = simg3.mean()

li = np.zeros((h/4,w/4))
li2 = np.zeros((h/4,w/4))
li,li2=cv.integral2(limg3)
si = np.zeros((h/4,w/4))
si2 = np.zeros((h/4,w/4))
si,si2=cv.integral2(simg3)


ha = h/4
wa = w/4

ls = li[0:ha-4,0:wa-4]+li[4:ha,4:wa]-li[4:ha,0:wa-4]-li[0:ha-4,4:wa]
ls2 = np.square(ls)
lss = li2[0:ha-4,0:wa-4]+li2[4:ha,4:wa]-li2[4:ha,0:wa-4]-li2[0:ha-4,4:wa]
test = np.square(limg3[2:ha-2,2:wa-2]-ls)/(ls2-lss)

ss = si[0:ha-4,0:wa-4]+si[4:ha,4:wa]-si[4:ha,0:wa-4]-si[0:ha-4,4:wa]
ss2 = np.square(ss)
sss = si2[0:ha-4,0:wa-4]+si2[4:ha,4:wa]-si2[4:ha,0:wa-4]-si2[0:ha-4,4:wa]
stest = np.square(simg3[2:ha-2,2:wa-2]-ss)/(ss2-sss)

test2 = np.abs(np.log(test))
stest2 = np.abs(np.log(stest))
cv.imshow("test",test2*10)
cv.imshow("stest",stest2*10)
input()

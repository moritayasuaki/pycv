#!/usr/bin/env python
# coding:utf-8
from scipy import ndimage
import os
import sys
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import scipy.cluster as cluster
import scipy.fftpack as fftpack
import numpy as np 
import json
import glob
from skimage import measure
from skimage import draw
from skimage import morphology
from skimage.morphology import watershed, disk, remove_small_objects
from skimage import color
from skimage import data
from skimage.filter import rank
from skimage import filter
from skimage import exposure
from skimage import transform
from skimage import feature
from skimage.util import img_as_ubyte, img_as_float
from skimage import io

debug = False


# color k-mean
def colormeans(img,labeled_img,nlabel):
  c = np.empty(nlabel)
  for m in np.arange(nlabel):
    c[m] = img[labeled_img == m].mean()
  return c

def line(img) :
  return (img - morphology.binary_erosion(img,disk(1)))

def merge(labels,code):
  ms = np.zeros_like(labels)
  for i,c in enumerate(code):
    ms += c * (labels == i)
  return ms

def norm(p0,p1) :
  y0,x0 = p0
  y1,x1 = p1
  return ((y0-y1)*(y0-y1) + (x0-x1)*(x0-x1))

def findminpos(obj0,obj1) :
  mp0 = []
  mp1 = []
  minnorm = 10000000000000000000000000
  ps0 = np.transpose(line(obj0).nonzero())
  ps1 = np.transpose(line(obj1).nonzero())
  for p0 in ps0 :
    for p1 in ps1 :
      if (norm(p0,p1) < minnorm) :
        minnorm = norm(p0,p1)
        mp0 = p0
        mp1 = p1
  return (mp0,mp1,minnorm)


def run(image):

# adjustment color of image
  image = exposure.adjust_gamma(image,gamma=0.4)
  image = img_as_ubyte(image)

# denoise image
  denoised = rank.median(image, disk(5))

# find continuous region (low gradient) --> markers

  height = len(image[0,:])
  width = len(image[:,0])

  size = height * width

  slen = np.sqrt(size)

  markers = rank.gradient(denoised, disk(slen*0.004)) < 10
  markers = remove_small_objects(markers,size*0.0005)
  markers, nmarks = ndimage.label(markers)

#local gradient
  gradient = rank.gradient(denoised, disk(1))

# process the watershed
  seg = watershed(gradient, markers) - 1


  cs = colormeans(image,seg,nmarks)
  k = 4

  center, dist = cluster.vq.kmeans(cs,k)
  scenter = np.sort(center)
  code, distance = cluster.vq.vq(cs,scenter)


  mseg = merge(seg,code)

  electroads,n = ndimage.label(mseg == 0)
  trigger = mseg == 1
  tube = mseg != 3

  if debug:
    plt.imshow(electroads+tube)
    plt.show()


  t0 = tube
  t1 = measure.block_reduce(t0,(2,2),func=np.min) # 512
  t2 = measure.block_reduce(t1,(2,2),func=np.min) # 256
  t3 = measure.block_reduce(t2,(2,2),func=np.min) # 128
  t3 = morphology.binary_opening(t3,disk(2))

  h3,w3 = t3.shape
  r = np.sqrt(h3*w3)

  rads = np.arange(int(r/4),int(r/2))

  houghs = transform.hough_circle(line(t3), rads)

  mri,my,mx = np.unravel_index(np.argmax(houghs),houghs.shape)
  mr = rads[mri]

# circle = draw.circle_perimeter(my,mx,rads[mr])
# draw.set_color(t3, circle, 2)
  d3 = morphology.binary_dilation(t3,disk(1))
  d2 = ndimage.zoom(d3,(2,2)) * t2

  rads = np.arange(2*(mr-2),2*(mr+2));
  houghs = transform.hough_circle(line(d2), rads)[:,2*(my-2):2*(my+2),2*(mx-2):2*(mx+2)]

  tri,tmy,tmx = np.unravel_index(np.argmax(houghs),houghs.shape)
  mx = 2*(mx-2)+tmx
  my = 2*(my-2)+tmy
  mr = rads[tri]

  d2 = morphology.binary_dilation(t2,disk(1))
  d1 = ndimage.zoom(d2,(2,2)) * t1

  rads = np.arange(2*(mr-2),2*(mr+2));
  houghs = transform.hough_circle(line(d1), rads)[:,2*(my-2):2*(my+2),2*(mx-2):2*(mx+2)]

  tri,tmy,tmx = np.unravel_index(np.argmax(houghs),houghs.shape)
  mx = 2*(mx-2)+tmx
  my = 2*(my-2)+tmy
  mr = rads[tri]

  d2 = morphology.binary_dilation(t1,disk(1))
  d0 = ndimage.zoom(d1,(2,2)) * t0

  rads = np.arange(2*(mr-2),2*(mr+2));
  houghs = transform.hough_circle(line(d0), rads)[:,2*(my-2):2*(my+2),2*(mx-2):2*(mx+2)]

  tri,tmy,tmx = np.unravel_index(np.argmax(houghs),houghs.shape)
  mx = 2*(mx-2)+tmx
  my = 2*(my-2)+tmy
  mr = rads[tri]

  circle = draw.circle_perimeter(my,mx,mr)



  el00 = electroads==1
  el10 = electroads==2
  el01 = measure.block_reduce(el00,(2,2),func=np.min) # 512
  el11 = measure.block_reduce(el10,(2,2),func=np.min)
  el02 = measure.block_reduce(el01,(2,2),func=np.min) # 256
  el12 = measure.block_reduce(el11,(2,2),func=np.min)
  el03 = measure.block_reduce(el02,(2,2),func=np.min) # 128
  el13 = measure.block_reduce(el12,(2,2),func=np.min)

  mp0,mp1,_ = findminpos(el03,el13)
  y0,x0 = mp0
  y1,x1 = mp1
  cir0 = np.zeros_like(el02)
  cir1 = np.zeros_like(el12)
  draw.set_color(cir0,draw.circle(2*y0,2*x0,7),1)
  draw.set_color(cir1,draw.circle(2*y1,2*x1,7),1)
  mp0,mp1,_ = findminpos(el02 * cir0, el12 * cir1)
  y0,x0 = mp0
  y1,x1 = mp1
  cir0 = np.zeros_like(el01)
  cir1 = np.zeros_like(el11)
  draw.set_color(cir0,draw.circle(2*y0,2*x0,7),1)
  draw.set_color(cir1,draw.circle(2*y1,2*x1,7),1)
  mp0,mp1,_ = findminpos(el01 * cir0, el11 * cir1)
  y0,x0 = mp0
  y1,x1 = mp1
  cir0 = np.zeros_like(el00)
  cir1 = np.zeros_like(el10)
  draw.set_color(cir0,draw.circle(2*y0,2*x0,20),1)
  draw.set_color(cir1,draw.circle(2*y1,2*x1,20),1)
  mp0,mp1,minnorm = findminpos(el00 * cir0, el10 * cir1)
  y0,x0 = mp0
  y1,x1 = mp1

  mline = draw.line(y0,x0,y1,x1)
  cimage = color.gray2rgb(image)

  draw.set_color(cimage, circle, (0,255,0))
  draw.set_color(cimage, mline, (255,0,0))

  cimage[y0,x0] = (255,255,255)
  cimage[y1,x1] = (255,255,255)
  d = np.sqrt(minnorm)/mr

  if debug:
    io.imshow(cimage)
    io.show()

  return (cimage,d,mr,(my,mx),(y0,x0),(y1,x1))

def rundir(path):
  jpgs = glob.glob(path + '/*.jpg')
  datalist = [];
  for jpg in jpgs :
    fname = os.path.basename(jpg)
    image = io.imread(jpg,as_grey=True)
    cimage,rel,r,center,p0,p1 = run(image)
    data = { 'fname':fname,
             'width':image.shape[1],
             'height':image.shape[0],
             'rel':rel,
             'r':r,
             'cx':center[1],
             'cy':center[0],
             'px0':p0[1],
             'py0':p0[0],
             'px1':p1[1],
             'py1':p1[0] }
    datalist.append(data)
    js = json.dumps(datalist)
    fd = open(path + '/result.json','w')
    fd.write(js)
    fd.close()
  return

dirname = sys.argv[1]
rundir(dirname)

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
from skimage.morphology import watershed, disk, remove_small_objects, square
from skimage import color
from skimage import data
from skimage.filter import rank
from skimage import filter
from skimage import exposure
from skimage import transform
from skimage import feature
from skimage.util import img_as_ubyte, img_as_float
from skimage import io
from scipy import stats

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


def flat(image,axis,a) :
  fimg = img_as_float(image)
  fs = np.mean(fimg,axis) 
  iss = np.arange(len(fs))
  slp, intc, _, _, _= stats.linregress(iss,fs)
  dss = iss * slp
  dimg = fimg - a * np.outer(np.ones(fimg.shape[1]),dss)
  dimg = exposure.rescale_intensity(dimg,in_range = (np.min(dimg),np.max(dimg)),out_range =(0.0,1.0))
  return dimg


def enhance(image,lthresh) :
  image = denoise(image,1)
  image = img_as_float(image)
  image2 = exposure.adjust_sigmoid(image,(lthresh[0]+lthresh[1])/2, gain = 5/(lthresh[1]-lthresh[0]))
  image3 = exposure.adjust_sigmoid(image,(lthresh[2]+lthresh[3])/2, gain = 5/(lthresh[3]-lthresh[2]))
  return (image2 + image3)/2

def denoise(image,n):
  sq = square(3)
  denoised = image
  for i in np.arange(3):
    denoised = rank.median(denoised,sq)
  return img_as_float(denoised)

def run(image):
  fimage = flat(image,0,0.35)
  height = len(image[0,:])
  width = len(image[:,0])

  size = height * width
  slen = np.sqrt(size)
  # denoise image
  denoised = denoise(fimage,1)
  cumsums,levels = exposure.cumulative_distribution(denoised,nbins=256)
  nthresh = [0.05,0.15,0.6,0.8]
  lthresh = []
  for i,cumsum in enumerate(cumsums):
    if cumsum >= nthresh[0] :
      lthresh.append(levels[i])
      del nthresh[0]
      if nthresh == [] :
        break
  lthresh = np.array(lthresh)

  low = denoised < lthresh[0]
  low = remove_small_objects(low,100)
  mid = (denoised >= lthresh[1]) * (denoised < lthresh[2])
  mid = remove_small_objects(mid,100)
  high = (denoised >= lthresh[3])
  high = remove_small_objects(high,100)
  high[0,:] = 1
  high[-1,:] = 1
  high[:,0] = 1
  high[:,-1] = 1
  enhanced = enhance(denoised,lthresh)
  # local gradient
  gradient = rank.gradient(enhanced, disk(5))
  if debug:
    plt.subplot(2,2,1)
    plt.plot(levels,cumsums)
    cumsums, levels = exposure.cumulative_distribution(enhanced)
    plt.subplot(2,2,2)
    plt.plot(levels,cumsums)
    plt.subplot(2,2,3)
    plt.imshow(denoised)
    plt.subplot(2,2,4)
    plt.imshow(enhanced)
    plt.show()
    plt.imshow(gradient)
    plt.show()
    return
  menhanced = filter.gaussian_filter(enhanced,5)
  grad = rank.gradient(menhanced,disk(3))
  l = morphology.remove_small_objects(grad < np.percentile(grad,20))
  t,n = ndimage.label(l)
  # process the watershed
  # mseg = watershed(gradient, t) - 1
  mseg = watershed(gradient, low + mid*2 + high*3) - 1
  # if debug:
    # cimg = color.gray2rgb(image)
    # clabel = color.label2rgb(mseg)
    # plt.imshow(t)
    # plt.subplot(1,2,1)
    # plt.imshow(clabel)
    # plt.subplot(1,2,2)
    # plt.show()
    # return
  electroads,nelectroad = ndimage.label(mseg == 0)
  
  tube = mseg == 2


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

  el00 = []
  el10 = []
  if nelectroad < 2:
    print "error!"
    return (image,(1.0*width/2)/(1.0*(slen/3)),slen/3,(height/2,width/2),(height/2,width/4),(height/2,(3*width)/4))
  elif nelectroad == 2:
    el00 = electroads==1
    el10 = electroads==2
  else:
    el00s = 0
    el10s = 0
    for i in np.arange(nelectroad):
      t = i+1
      elt = electroads == t
      temps = np.sum(elt)
      if el00s < temps :
        el10s = el00s
        el00s = temps
        el10 = el00
        el00 = elt
      elif el10s < temps :
        el10s = temps
        el10 = elt


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
  if debug :
    plt.subplot(1,2,1)
    plt.imshow(mseg)
    plt.subplot(1,2,2)
    plt.imshow(cimage)
    plt.show()
    return

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
    obj = { 'list': datalist,
            'complete': False }
    js = json.dumps(obj)
    fd = open(path + '/result.json','w')
    fd.write(js)
    fd.close()
  obj = { 'list': datalist,
          'complete': True }
  js = json.dumps(obj)
  fd = open(path + '/result.json','w')
  fd.write(js)
  fd.close()
  return

dirname = sys.argv[1]
rundir(dirname)

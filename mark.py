#!/usr/bin/env python
from scipy import ndimage
import sys
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import scipy.cluster as cluster
import scipy.fftpack as fftpack
import numpy as np 
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

debug = True

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

def denoise(image,n):
  sq = square(3)
  denoised = image
  for i in np.arange(3):
    denoised = rank.median(denoised,sq)
  return denoised


def run(image):
  image = img_as_ubyte(image)
  height = len(image[0,:])
  width = len(image[:,0])
  size = height * width
  slen = np.sqrt(size)
  # denoise image
  image = img_as_ubyte(image)
  denoised = denoise(image,2)
  cumsums,levels = exposure.cumulative_distribution(denoised)
  nthresh = [0.08,0.15,0.7,0.8]
  lthresh = []
  for i,cumsum in enumerate(cumsums):
    if cumsum >= nthresh[0] :
      lthresh.append(levels[i])
      del nthresh[0]
      if nthresh == [] :
        break
  lthresh = np.array(lthresh)

  low = denoised < lthresh[0]
  low = remove_small_objects(low,20)
  mid = (denoised >= lthresh[1]) * (denoised < lthresh[2])
  mid = remove_small_objects(mid,20)
  high = (denoised >= lthresh[3])
  high = remove_small_objects(high,20)
  high[0,:] = 1
  high[-1,:] = 1
  high[:,0] = 1
  high[:,-1] = 1
  eimg = exposure.rescale_intensity(denoised,in_range=(lthresh[0],lthresh[3]))
  harris = feature.corner_kitchen_rosenfeld(eimg)
  if debug:
    io.imshow(harris)
    io.show()
  # we can process adjust_gamma for eimg

  # edenoised = rank.median(eimg, square(10)) # non-obvious parameter
  # gr = rank.gradient(denoised, square(slen*0.01)) # non-obvious parameter
  # splitter = gr <= 4
  # if debug:
    # io.imshow(splitter)
    # io.show()
    # return

  # find continuous region (low gradient) --> markers

  # low*=splitter
  # low=remove_small_objects(low,size*0.00005)
  # mid*=splitter
  # mid=remove_small_objects(mid,size*0.00005)
  # high*=splitter
  # high=remove_small_objects(high,size*0.0005)

  # local gradient
  gradient = rank.gradient(eimg, square(10))
  t,n = ndimage.label(low + mid + high)
  if debug:
    io.imshow(eimg)
    io.show()
    io.imshow(low + 2*mid + 3*high)
    io.show()
  # process the watershed
  # mseg = watershed(gradient, t) - 1
  mseg = watershed(gradient, low + mid*2 + high*3) - 1
  if debug:
    plt.imshow(mseg)
    plt.show()
    return

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
    return "error"
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

  if debug:
    plt.imshow(tube*100+image)
    plt.show()

  return (cimage,d,mr,(my,mx),(y0,x0),(y1,x1))

# loading image
if debug:
  fnames = ["./data/S12058-8_t.jpg","./data/S12057-1_s.jpg","./data/S12057-3_t.jpg"]
else:
  fnames = sys.argv[1:]

for fname in fnames:
  image = io.imread(fname,as_grey=True)
  if debug:
    imageu = image +np.random.standard_normal(image.shape)*0.05
    imageu = exposure.rescale_intensity(imageu,in_range=(0.0,1.0),out_range=(0.2,1.0))
    imageu = exposure.adjust_gamma(imageu,0.4)
    run(image)
    run(imageu)
    continue
  cimage,d,r,center,p0,p1 = run(image)
  io.imsave(fname+".png",cimage)
  f = open(fname+".txt","w")
  f.write(fname+", ")
  f.write(str(d)+", ")
  f.write(str(r)+", ")
  f.write(str(center)+", ")
  f.write(str(p0)+", ")
  f.write(str(p1)+"\n")
  f.close()
  print (fname + " finished.") 

from scipy import ndimage
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import scipy.cluster as cluster
import scipy.fftpack as fftpack
import numpy as np 
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


# loading image
image = io.imread("./data/S12057-1_s.jpg",as_grey=True)
image2 = io.imread("./data/S12057-1_s.jpg",as_grey=True)

# adjustment color of image
image = exposure.adjust_gamma(image,gamma=0.5)
image = img_as_ubyte(image)

# denoise image
denoised = rank.median(image, disk(5))

# find continuous region (low gradient) --> markers

height = len(image[0,:])
width = len(image[:,0])

size = height * width

slen = np.sqrt(size)

markers = rank.gradient(denoised, disk(slen*0.002)) < 7
markers = remove_small_objects(markers,size*0.00005)
markers, nmarks = ndimage.label(markers)

#local gradient
gradient = rank.gradient(denoised, disk(1))

# process the watershed
seg = watershed(gradient, markers) - 1

# color k-mean
def colormeans(img,labeled_img,nlabel):
  c = np.empty(nlabel)
  for m in np.arange(nlabel):
    c[m] = img[labeled_img == m].mean()
  return c

cs = colormeans(image,seg,nmarks)
k = 4

center, dist = cluster.vq.kmeans(cs,k)
scenter = np.sort(center)
code, distance = cluster.vq.vq(cs,scenter)

def merge(labels,code):
  ms = np.zeros_like(labels)
  for i,c in enumerate(code):
    ms += c * (labels == i)
  return ms

mseg = merge(seg,code)

electroads,n = ndimage.label(mseg == 0)
trigger = mseg == 1
tube = mseg != 3

plt.imshow(electroads+tube)
plt.show()

def line(img) :
  return (img - morphology.binary_erosion(img,disk(1)))

t0 = line(tube)
t1 = measure.block_reduce(t0,(2,2),func=np.max)
t2 = measure.block_reduce(t1,(2,2),func=np.max)
t3 = measure.block_reduce(t2,(2,2),func=np.max)
t4 = measure.block_reduce(t3,(2,2),func=np.max)
io.imshow(t4)
io.show()

peak = 0
rs = np.arange(10,30)
hough_res = transform.hough_circle(t4, rs)
for i,r in enumerate(rs):
  tmp = feature.peak_local_max(hough_res[i],num_peaks=1)[0]
  if (peak < hough_res[i][tmp[0],tmp[1]]):
    peak = hough_res[i][tmp[0],tmp[1]]
    peakpos = tmp
    peakr = r

peak = 0
rs = np.arange(peakr*2-3,peakr*2+3)
hough_res = transform.hough_circle(t3, rs)
for i,r in enumerate(rs):
  tmp = feature.peak_local_max(hough_res[i],num_peaks=1)[0]
  if (peak < hough_res[i][tmp[0],tmp[1]]):
    peak = hough_res[i][tmp[0],tmp[1]]
    peakpos = tmp
    peakr = r

peak = 0
rs = np.arange(peakr*2-3,peakr*2+3)
hough_res = transform.hough_circle(t2, rs)
for i,r in enumerate(rs):
  tmp = feature.peak_local_max(hough_res[i],num_peaks=1)[0]
  if (peak < hough_res[i][tmp[0],tmp[1]]):
    peak = hough_res[i][tmp[0],tmp[1]]
    peakpos = tmp
    peakr = r

peak = 0
rs = np.arange(peakr*2-3,peakr*2+3)
hough_res = transform.hough_circle(t1, rs)
for i,r in enumerate(rs):
  tmp = feature.peak_local_max(hough_res[i],num_peaks=1)[0]
  if (peak < hough_res[i][tmp[0],tmp[1]]):
    peak = hough_res[i][tmp[0],tmp[1]]
    peakpos = tmp
    peakr = r

peak = 0
rs = np.arange(peakr*2-3,peakr*2+3)
hough_res = transform.hough_circle(t0, rs)
for i,r in enumerate(rs):
  tmp = feature.peak_local_max(hough_res[i],num_peaks=1)[0]
  if (peak < hough_res[i][tmp[0],tmp[1]]):
    peak = hough_res[i][tmp[0],tmp[1]]
    peakpos = tmp
    peakr = r

plt.imshow(tube)
plt.show()
plt.imshow(t0)
plt.show()
center_y, center_x = peakpos

cx, cy = draw.circle_perimeter(center_y, center_x, peakr)
out = color.gray2rgb(image)
out[cy, cx] = (220,20,20)

plt.imshow(out)
plt.show()

# for i in range(0,25):
  # tube2 = morphology.binary_erosion(tube2,morphology.square(2))

# tube_edge = morphology.binary_dilation(tube2,disk(1)) != tube2

# plt.imshow(tube2)
# plt.show()


# d = pcorr(tube2,tube2[::-1,::-1])
# dy, dx = ndimage.measurements.maximum_position(d)
# if (dy > height/2):
  # dy -= height
# if (dx > width/2):
  # dx -= width
# cy = height/2 + dy
# cx = height/2 + dx

# circ = draw.circle_perimeter(cy,cx,10)
# image[circ] = 1
# plt.imshow()
# plt.show()

# hough_radii = np.arange(int(tube_len*0.5),int(tube_len*1.0),5)
# hough_res = transform.hough_circle(tube_edge, hough_radii)

# centers = []
# accums= []
# radii = []

# out = color.gray2rgb(image)

# # for radius, h in zip(hough_radii, hough_res):
  # # peaks = feature.peak_local_max(h, num_peaks=2)
  # # centers.extend(peaks)
  # # accums.extend(h[peaks[:,0],peaks[:,1]])
  # # radii.extend([radius, radius])

# print centers

# for idx in np.argsort(accums)[::-1][:1]:
  # center_x, center_y = centers[idx]
  # radius = radii[idx]
  # cx, cy = draw.circle_perimeter(center_y, center_x, radius)
  # out[cy, cx] = (220,20,20)

# plt.imshow(out, cmap=plt.cm.gray)
# plt.show()

# display results
# fig, axes = plt.subplots(ncols=4, figsize=(8, 2.7))
# ax0, ax1, ax2, ax3 = axes

# ax0.imshow(image, cmap=plt.cm.gray, interpolation='nearest')
# ax1.imshow(gradient, cmap=plt.cm.spectral, interpolation='nearest')
# ax2.imshow(markers, cmap=plt.cm.spectral, interpolation='nearest')
# ax3.imshow(image, cmap=plt.cm.gray, interpolation='nearest')
# ax3.imshow(seg, cmap=plt.cm.spectral, interpolation='nearest', alpha=1)

# for ax in axes:
    # ax.axis('off')

# plt.subplots_adjust(hspace=0.01, wspace=0.01, top=1, bottom=0, left=0, right=1)
# plt.show()


from scipy import ndimage
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import numpy as np 
from skimage.morphology import watershed, disk, remove_small_objects
from skimage import data
from skimage.filter import rank
from skimage import filter
from skimage import exposure
from skimage.util import img_as_ubyte
from skimage import io

image = io.imread("./data/S12057-3_s.jpg",as_grey=True)
image = exposure.adjust_gamma(image,gamma=0.5)
image = img_as_ubyte(image)

# denoise image
denoised = rank.median(image, disk(5))

# find continuous region (low gradient) --> markers
markers = rank.gradient(denoised, disk(int(len(image)/256))) < 10
markers = remove_small_objects(markers,1000)
markers = ndimage.label(markers)[0]

#local gradient
gradient = rank.gradient(denoised, disk(1))
# sobel = filter.sobel(denoised)

# process the watershed
seg = watershed(gradient, markers)

# display results
fig, axes = plt.subplots(ncols=4, figsize=(8, 2.7))
ax0, ax1, ax2, ax3 = axes

ax0.imshow(image, cmap=plt.cm.gray, interpolation='nearest')
ax1.imshow(gradient, cmap=plt.cm.spectral, interpolation='nearest')
ax2.imshow(markers, cmap=plt.cm.spectral, interpolation='nearest')
# ax3.imshow(image, cmap=plt.cm.gray, interpolation='nearest')
ax3.imshow(seg, cmap=plt.cm.spectral, interpolation='nearest', alpha=1)

for ax in axes:
    ax.axis('off')

plt.subplots_adjust(hspace=0.01, wspace=0.01, top=1, bottom=0, left=0, right=1)
plt.show()


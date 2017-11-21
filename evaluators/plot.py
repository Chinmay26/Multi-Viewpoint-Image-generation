#from matplotlib.pyplot import imshow, show
from matplotlib import pyplot as plt
import cv2
import numpy as np


def plot(images):
	#fig = plt.figure()
	total = len(images)
	#imgplot = plt.imshow(lum_img)
	#fig, axes = plt.subplots(nrows=total, ncols=1, sharex=True, sharey=False)
	fig, axes = plt.subplots(total,1)

	print(type(axes))
	#ax1 = None
	#for i in range(total):
	#	ax1 = fig.add_subplot(total,1,i+1)
	#	ax1.plot(images[i])
	for i in range(total):
		print(images[i].shape, "IMG DIM", np.max(images[i]), np.min(images[i]))
		axes[i].imshow(images[i])
		#cv2.imshow('img',images[i])
		#cv2.waitKey(0)


		#plt.imshow(images[i])
	plt.show()
	#imshow()
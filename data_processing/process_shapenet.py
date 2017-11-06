import os, sys
import numpy as np
from scipy import misc
from random import shuffle
from copy import deepcopy

class Shapenet(object):
	def __init__(self, folder_path):
		self.path = folder_path

	""" Load images from the directory path.
		Assumes images are of the format imgid_azimuthangle_elevationangle. Ex: 1we2wd_0_0
	"""
	def get_dataset(self, train_split=0.8, azimuth_angles=range(0,360,10), elevation_angle=[0]):
		images, test, train = {}, {}, {}

		#Read all the images
		for subdir, dirs, files in os.walk(self.path):
			for fname in files:
				if fname.endswith('.png'):
					name,extension = os.path.splitext(fname)
					fpart, elevation_angle = name.rsplit('_',1)
					img_id, azimuth_angle = fpart.rsplit('_', 1)

					fp = os.path.join(self.path, subdir, fname)
					img_data = misc.imread(fp).astype(np.float32)
					pos = int(azimuth_angle)

					if images.get(img_id):
						if images[img_id].get(elevation_angle):
							#images[img_id][elevation_angle].append(img_data)
							images[img_id][elevation_angle][pos] = img_data

						else:
							#images[img_id][elevation_angle] = [img_data]
							images[img_id][elevation_angle] = [0] * 36 #Fix later
							images[img_id][elevation_angle][pos] = img_data
					else:
						images[img_id] = {}
						#images[img_id][elevation_angle] = [img_data]
						images[img_id][elevation_angle] = [0] * 36 #Fix later
						images[img_id][elevation_angle][pos] = img_data

		#Split training and test sets
		'''
		img_ids = images.keys()
		shuffle(img_ids)
		l = len(img_ids)
		train_len = int(l * train_split)
		test_len = l - train_len

		train, test = {}, {}
		train = deepcopy(images)
		for i in range(train_len,l):
			test[img_ids[i]] = images[images[i]]
			del train[img_ids[i]]

		assert((len(train.keys()) + len(test.keys())) == l)
		return train,test
		'''
		print(len(images.keys()), len(images.values()))
		val = (images.values()) # contains multiple elevation angles
		image_data = []
		for v in val:
			#print(type(v.values()))
			image_data.extend(v['0']) # Fix this later	
		#shuffle(image_data)
		l = len(image_data)
		k = int(len(images.keys()) * train_split) * 36 # Fix later
		train = np.array(image_data[:int(k)])
		test = np.array(image_data[int(k):])

		return train,test



if __name__=='__main__':
	dpath = '/home/chinmay/CODE/deep_learning/shapenet_datasets/mug/models/3dw/'
	s = Shapenet(dpath)
	train, test = s.get_dataset()

	train_labels = np.zeros(train.shape)

	for cnt in range(train.shape[0] // 36):
		train_labels[cnt*36:(cnt+1)*36] = np.roll(train[cnt*36:(cnt+1)*36], 1, axis=0)

	batch_target = np.roll(train[:,:36,:,:], 1, axis=1)

	from matplotlib import pyplot as plt
	#print(train[0][...,-1] == batch_target[0][...,-1])
	print(train.shape)
	print(np.array_equal(batch_target, train))

	fig, axes = plt.subplots(2,2)

	axes[0][0].imshow(train[110][...,-1])
	axes[0][1].imshow(train_labels[110][...,-1])
	axes[1][0].imshow(train[112][...,-1])
	axes[1][1].imshow(train_labels[112][...,-1])
	#ax1.plot(train[0][...,-1])

	#ax2 = fig.add_subplot(212)
	#ax2.plot([(7, 2), (5, 3)], [(1, 6), (9, 5)])

	plt.show()
import os, sys
from scipy import misc
from random import shuffle
from copy import deepcopy

class Shapenet(object):
	def __init__(self, folder_path):
		self.path = folder_path

	""" Load images from the directory path.
		Assumes images are of the format imgid_azimuthangle_elevationangle. Ex: 1we2wd_0_0
	"""
	def get_data(self, train_split=0.8, azimuth_angles=range(0,360,10), elevation_angle=[0]):
		images, test, train = {}, {}, {} # {'id': [images]}

		#Read all the images
		for subdir, dirs, files in os.walk(self.path):
			for fname in files:
				if fname.endswith('.png'):
					name = os.path.splitext(fname)
					fpart, elevation_angle = name.rsplit('_',1)
					img_id, azimuth_angle = fpart.rsplit('_', 1)

					if images.get(img_id) is not None:
						fp = os.path.join(self.path, subdir, fname)
						images[img_id][elevation_angle] = misc.imread(fp).astype(np.float32)

		#Split training and test sets
		img_ids = images.keys()
		shuffle(img_ids)
		l = len(img_ids)
		train_len = int(l * train_split)
		test_len = l - test_len

		train, test = {}, {}
		train = deepcopy(images)
		for i in range(train_len,l):
			test[img_ids[i]] = images[images[i]]
			del train(img_ids[i])

		assert((len(train.keys()) + len(test.keys())) == l)
		return train,test

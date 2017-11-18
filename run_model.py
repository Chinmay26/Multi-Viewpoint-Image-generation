import argparse, json, sys
import tensorflow as tf
from trainers import trainer
from models import auto_encoder
from data_processing import process_shapenet
from evaluators import plot
from utils import helper
import numpy as np


import matplotlib
from matplotlib.pyplot import imshow, show
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

def run(config_file):
	config = {}
	with open(config_file) as cf:
		config = json.load(cf)

	tf.reset_default_graph()
	cur_sess = tf.Session()
	config['trainer']['batch_size'] = config['batch_size']
	config['model']['batch_size'] = config['batch_size']
	config['trainer']['session'] = cur_sess

	print("Getting dataset")
	#Get training and test split
	dpath = '/home/chinmay/CODE/deep_learning/shapenet_datasets/mug/models/3dw/'
	s = process_shapenet.Shapenet(dpath)
	train, test = s.get_dataset()
	print("Created test/ train split")


	train = helper.input_normalization(train)
	test = helper.input_normalization(test)

	'''
	plot.plot([train[0], train[1], train[2]])
	train = helper.input_normalization(train)
	test = helper.input_normalization(test)

	result = helper.denormalization(np.array([train[0], train[1], train[2]]))
	print()
	plot.plot([result[0], result[1], result[2]])
	'''


	print("=====Creating model=====")
	#if config['model'] == 'encoder':
	model = auto_encoder.AutoEncoder(config['model'])
	model.build()
	print("======Done model======")

	print("=====Creating trainer=====")
	t = trainer.Trainer(config['trainer'], train, test)
	t.setup_graph(model)
	print("=====Done trainer=====")
	

	print("======Starting training:======")
	t.train_model()
	print("======Training Done======")

	print("======Starting test:======")
	result = t.test_model()
	print("======Testing Done======")


	result = helper.denormalization(result)
	test = helper.denormalization(test)

	#im = [test[5][...,-1], result[0][5][...,-1]]
	im = [test[5], result[0][5]]
	#for i in range(3):
	#	im.append(result[i][...,::-1])

	#print(test[0][...,-1].shape, "TEST")
	#print(result[0][...,-1].shape, "RESULT")
	plot.plot(im)
	im = [np.uint8(test[5]), np.uint8(result[0][5])]
	plot.plot(im)
	#print(result[0][0][...,::-1].shape)

	

if __name__=='__main__':
	if(len(sys.argv) < 2):
		print("Expects a config path file as input")
		raise Exception("Expects a config file as a argument")
	run(sys.argv[1])
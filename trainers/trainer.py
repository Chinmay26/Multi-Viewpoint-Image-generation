from models import auto_encoder
import tensorflow.contrib.layers as layers
import tensorflow as tf
import numpy as np
from evaluators import plot
from utils import helper

class Trainer(object):
	def __init__(self, config, train, test):
		self.batch_size = config.get('batch_size') if config.get('batch_size') else 32
		self.learning_rate = config.get('learning_rate') if config.get('learning_rate') else 1e-4
		self.epochs = config.get('epochs') if config.get('epochs') else 50
		self.session = config.get('session')
		self.global_step = tf.train.get_or_create_global_step(graph=None)
		self.train = train
		self.test = test

	""" Construct the optimizer"""
	def build_optimizer(self, loss, gstep=None, lrate=None):
		gstep = self.global_step if gstep is None else gstep
		#lrate = self.learning_rate if lrate is None else lrate
		#self.opt = layers.optimize_loss(loss = loss, global_step=gstep, learning_rate=lrate, 
		#		optimizer=tf.train.AdamOptimizer)

		lr = tf.train.exponential_decay(5e-4, gstep, 1000, 0.96, staircase=True)
		optimizer = tf.train.AdamOptimizer(lr)
		self.opt = optimizer.minimize(loss=loss)

	def setup_graph(self, model):
		self.model = model
		self.loss = self.model.get_loss()
		self.build_optimizer(self.loss)
		self.session.run(tf.global_variables_initializer())

	""" Train the network """
	def train_model(self, is_train=True):
		"""
		loss = model.get_loss()
		self.build_optimizer(loss)
		dataset = self.train if is_train else self.test
		l = len(dataset)
		"""
		dataset = self.train
		l = len(dataset)
		print("Total iterations in epoch",int(l/self.batch_size))
		for e in range(self.epochs):
			for i in range(int(l/self.batch_size)):
			#for i in range(5):
				batch_input = dataset[i*self.batch_size: (i+1)*self.batch_size]
				#batch_target = np.roll(batch_input, 1, axis=1)

				batch_target = np.zeros(batch_input.shape)

				for cnt in range(batch_input.shape[0] // 36):
					batch_target[cnt*36:(cnt+1)*36] = np.roll(batch_input[cnt*36:(cnt+1)*36], 1, axis=0)
				
				'''
				k1 = batch_input[0]
				k2 = batch_target[0]
				print(batch_target[0])
				res = helper.denormalization([k1,k2])
				plot.plot(res)
				'''

				feed_dict = {self.model.image: batch_input, self.model.target_image: batch_target}

				computed_loss,computed_opt, gstep =  self.session.run([self.loss, self.opt, self.global_step], feed_dict=feed_dict)
				print("Epoch: {}/{}...".format(e+1, self.epochs), "Training loss: {:.4f}".format(computed_loss))


	def test_model(self):
		dataset = self.test

		l = len(dataset)
		print("Total iterations in epoch",int(l/self.batch_size))
		test_output = []
		for i in range(int(l/self.batch_size)):
		#for i in range(5):
			batch_input = dataset[i*self.batch_size: (i+1)*self.batch_size]
			train_labels = np.zeros(batch_input.shape)

			for cnt in range(batch_input.shape[0] // 36):
				train_labels[cnt*36:(cnt+1)*36] = np.roll(batch_input[cnt*36:(cnt+1)*36], 1, axis=0)
			
			feed_dict = {self.model.image: batch_input, self.model.target_image: train_labels}

			op, computed_loss =  self.session.run([self.model.model_output, self.loss], feed_dict=feed_dict)
			test_output.append(op)
			print("Training loss: {:.4f}".format(computed_loss))

		return test_output

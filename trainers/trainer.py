from models import auto_encoder
import tensorflow.contrib.layers as layers
import tensorflow as tf
import numpy as np
from evaluators import plot
from utils import helper

log_dir = "/home/chinmay/CODE/deep_learning/599-project/repo/Multi-Viewpoint-Image-generation/log_dir"

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
		lrate = self.learning_rate if lrate is None else lrate
		
		lr = tf.train.exponential_decay(lrate, gstep, 1000, 0.98, staircase=True)
		optimizer = tf.train.AdamOptimizer(lr)
		self.opt = layers.optimize_loss(loss = loss, global_step=gstep, learning_rate=lrate, 
				optimizer=optimizer)


		#self.opt = optimizer.minimize(loss=loss)

	def setup_graph(self, model):
		self.model = model
		self.loss = self.model.get_loss()
		self.build_optimizer(self.loss)
		self.session.run(tf.global_variables_initializer())

		self.summary_op = tf.summary.merge_all()
		self.summary_writer = tf.summary.FileWriter(log_dir, self.session.graph)

	""" Train the network """
	def train_model(self, is_train=True):
		"""
		loss = model.get_loss()
		self.build_optimizer(loss)
		dataset = self.train if is_train else self.test
		l = len(dataset)
		"""
		#for op in self.session.graph.get_operations(): 
		#	print(op.name)

		#v = [n.name for n in tf.get_default_graph().as_graph_def().node]
		#print(v)

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
				x_ = tf.reduce_max(self.model.model_output)

				feed_dict = {self.model.image: batch_input, self.model.target_image: batch_target}

				#<<<<<<< Updated upstream
				computed_loss,computed_opt, gstep, summary =  self.session.run([self.loss, self.opt, self.global_step, self.summary_op], feed_dict=feed_dict)
				#=======
				'''
				#computed_loss,computed_opt, output,gstep,  c1, c2, c3, d1, d2, d3, x_ \
				#=  self.session.run([self.loss, self.opt, self.model.model_output, self.global_step, self.model.c1, self.model.c2, self.model.c3, self.model.d1, self.model.d2,self.model.d3, x_], feed_dict=feed_dict)
				
				computed_loss,computed_opt, output,gstep, d2 \
				=  self.session.run([self.loss, self.opt, self.model.model_output, self.global_step, self.model.d2], feed_dict=feed_dict)
				
				#print("c1",  np.max(c1))
				#print("c2",  np.max(c2))
				#print("c3",  np.max(c3))
				#print("d1",  np.max(d1))
				print("d2",  np.max(d2), np.min(d2))
				#print("d3",  np.max(d3))
				#print("x_",  np.max(x_))
				#print("optimiser",  self.opt)


				#>>>>>>> Stashed changes
				'''
				print("Epoch: {}/{}...".format(e+1, self.epochs), "Training loss: {:.4f}".format(computed_loss))

				if i % 10 == 0:
					self.summary_writer.add_summary(summary, global_step=gstep)


	def test_model(self):
		dataset = self.test

		l = len(dataset)
		print("Total iterations in epoch",int(l/self.batch_size))
		test_output = []
		#for i in range(int(l/self.batch_size)):
		for i in range(5):
			batch_input = dataset[i*self.batch_size: (i+1)*self.batch_size]
			train_labels = np.zeros(batch_input.shape)

			for cnt in range(batch_input.shape[0] // 36):
				train_labels[cnt*36:(cnt+1)*36] = np.roll(batch_input[cnt*36:(cnt+1)*36], 1, axis=0)
			
			feed_dict = {self.model.image: batch_input, self.model.target_image: train_labels}

			op, computed_loss =  self.session.run([self.model.model_output, self.loss], feed_dict=feed_dict)
			#print(tf.reduce_max(self.model.model_output), "TMAX")
			test_output.append(op)
			print("Training loss: {:.4f}".format(computed_loss))

		return test_output

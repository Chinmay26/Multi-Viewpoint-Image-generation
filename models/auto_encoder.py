from models import base_model
from utils import helper
import tensorflow as tf

class AutoEncoder(base_model.Model):
	def __init__(self,config):
		self.batch_size = config.get('batch_size') if config.get('batch_size') else 32
		self.image_height = config.get('height')
		self.image_width = config.get('width')
		self.channel = config.get('channel')

		self.image = tf.placeholder(shape=[self.batch_size, self.image_height, self.image_width, self.channel],
									dtype=tf.float32)
		self.target_image = tf.placeholder(shape=[self.batch_size, self.image_height, self.image_width, self.channel],
									dtype=tf.float32)


	def encoder(self, input_image, scope_name='encoder', reuse=False):
		with tf.variable_scope(scope_name):
			c1 = helper.conv2d(input_image, 16, kernel_h=3, kernel_w=3, k_stride=2, scope_name='conv1')
			c2 = helper.conv2d(c1, 32, kernel_h=3, kernel_w=3, k_stride=2, scope_name='conv2')
			c3 = helper.conv2d(c2, 64, kernel_h=3, kernel_w=3, k_stride=2, scope_name='conv3')
			return c3

	def decoder(self, dinput, scope_name='encoder'):
		with tf.variable_scope(scope_name):
			d1 = helper.deconv2d(dinput, kernel_size=3, stride=2, num_filter=32, act=tf.nn.relu, scope_name='dconv1')
			d2 = helper.deconv2d(d1, kernel_size=3, stride=2, num_filter=16, act=tf.nn.relu, scope_name='dconv2')
			d3 = helper.deconv2d(d2, kernel_size=3, stride=2, num_filter=3, act=tf.nn.tanh, scope_name='dconvv3')
			return d3

	def get_feed_dict(self, batch_input, batch_output):
		return {self.image: batch_input, self.target_image: batch_output}

	def get_loss(self):
		return self.loss

	def build(self):
		output = []
		ip_image = None
		target_image = None
		with tf.variable_scope('AutoEncoder'):
			e = self.encoder(self.image)
			d = self.decoder(e)
			self.model_output = d
			output.append(d)
		op1 = tf.stack(output, axis=0)
		op1 = tf.transpose(op1, [1, 0, 2, 3, 4])
		op2 = tf.expand_dims(self.target_image, axis=1)
		print(op1.get_shape(), "OP1")
		print(op2.get_shape(), "OP2")
		self.loss = tf.reduce_mean(tf.abs(op1 - op2))
		#self.loss = tf.reduce_mean(tf.abs(d - self.target_image))
		self.output = output
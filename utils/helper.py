import tensorflow as tf
import tensorflow.contrib.layers as layers
import tensorflow.contrib.slim as slim
import numpy as np

"""
	2D Convolution layer
	To-do : Add weights_init, bias_init... needed?
"""
def conv2d(input, output, kernel_h=3, kernel_w=3, k_stride=2, scope_name="conv2d", act=tf.nn.relu):
	#act = tf.nn.leaky_relu
	with tf.variable_scope(scope_name):
		return layers.conv2d(inputs=input, num_outputs=output, kernel_size=[kernel_h, kernel_w], stride=k_stride, activation_fn=act,
			 biases_initializer=tf.zeros_initializer(), weights_initializer=tf.contrib.layers.xavier_initializer())
   
"""
	2D DeConvolution layer
	To-do : Add weights_init, bias_init... needed?
"""
def deconv2d(input, kernel_size, stride, num_filter, scope_name='deconv2d'):
	with tf.variable_scope(scope_name): 
		stride_shape = [stride, stride]
		kernel_shape = [kernel_size, kernel_size]
		return layers.conv2d_transpose(inputs=input, num_outputs=num_filter, stride=stride_shape, kernel_size= kernel_shape,
			padding='SAME', biases_initializer=tf.zeros_initializer(), weights_initializer=tf.contrib.layers.xavier_initializer()
		)

"""
	Input Normalization
"""
def input_normalization(input, drange=[-1,1]):
	#m1 = np.max(input)
	#m2 = np.min(input)
	#r = np.max(drange) - np.min(drange)
	#return (np.array(input) / m1)*2 - np.max(drange)
	input *= 2
	input /= 255.0
	input -= 1
	return input


def input_0_normalization(input, drange=[-1,1]):
	#m1 = np.max(input)
	#m2 = np.min(input)
	#r = np.max(drange) - np.min(drange)
	#return (np.array(input) / m1)*2 - np.max(drange)
	input *= 2
	input /= 255.0
	input -= 1
	return input

"""
	Input De-Normalization
"""
def denormalization(input, drange=[0,255]):
	input = np.array(input)
	input += 1
	input *= 255
	input /= 2
	return np.rint(input)


def tanh_denormalization(input, drange=[-1,1]):
	input = np.array(input)
	input += 1
	input *= np.max(drange)
	input /= 2
	return input

"""
	Batch Normalization
"""
def batch_norm(input, is_train, activation_fn=None, scope_name="bn_act"):
	with tf.variable_scope(scope_name):
		return tf.contrib.layers.batch_norm(input, scale=True, updates_collections=None)
    
"""
	2D Max Pooling 
"""
def max_pool2d(input, kernel_size, stride, scope_name='max_pool2d'):
	with tf.variable_scope(scope_name):
		kernel_shape = [kernel_size, kernel_size]
		return layers.max_pool(input, ksize=ksize, strides=stride, padding='SAME')


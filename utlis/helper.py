import tensorflow as tf
import tensorflow.contrib.layers as layers
import tensorflow.contrib.slim as slim

"""
	2D Convolution layer
	To-do : Add weights_init, bias_init... needed?
"""
def conv2d(input, output_shape, kernel_h=3, kernel_w=3, k_stride=2, scope_name="conv2d", act=tf.nn.relu):
    with tf.variable_scope(scope_name):
        return slim.conv2d(inputs=input, num_outputs=output_shape, kernel_size=[kernel_h, kernel_w], stride=k_stride, activation_fn=act)
   
"""
	2D DeConvolution layer
	To-do : Add weights_init, bias_init... needed?
"""
def deconv2d(input, kernel_size, stride, num_filter, act=tf.nn.relu, scope_name='deconv2d'):
	with tf.variable_scope(scope_name): 
	    stride_shape = [stride, stride]
	    kernel_shape = [kernel_size, kernel_size]
	    return layers.conv2d_transpose(inputs=input, out_shape=num_filter, stride=stride_shape, kernel_size= kernel_shape)


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


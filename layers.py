import torch.nn as nn
import torch.nn.functional as F
import numpy as np


#def lrelu(x, rate=0.1):
	#does this need to be a torch fn or np is fine
	#return np.maximum(np.minimum(x * rate, 0), x)
	#return nn.BatchNorm2d()

def conv2d_bn_relu(num_inputs, num_outputs, kernel_size, stride, is_training=True):
	#num_inputs = channels in input image
	#num_outputs = channels produced by conv
	weight = kernal_size

	conv = nn.Conv2d(num_inputs, num_outputs, kernel_size, stride=1, padding=0, dilation=1,groups=1) 
	bn = F.BatchNorm2d(num_outputs)
	lrelu = F.LeakyReLU(0.2)
	
	conv = lrelu(bn(conv))

	return conv

def conv2d_trans_bn_relu():
	conv
	batch_norm
	lrelu
	pass


def conv2d_trans_bn():
	
	pass

def conv2d_bn_lrelu():
	#input: minibatch x in_channels x iH x iW
	#weight: out_channels x in_channels/groups x kH x kW =? kernel_size
	#pyTorch doesn't have number of out filters?
	"""weight = kernal_size
	
	conv = F.conv2d(inputs, weight, stride) 
	conv = F.BatchNorm2d(kernel_size)
	conv = F.LeakyReLU(0.2, is_training=is_training)"""
	return conv

def conv2d_trans_bn_lrelu():

	pass


def fc_bn_relu(num_inputs, num_outputs):
	fc = nn.Linear(num_inputs, num_outputs) #input feature size, output feature size
	fc.weight = nn.init.normal(fc.weight, mean=0, std=0.02)

	#not sure how to regularize the 
	#weights_regularizer=tf.contrib.layers.l2_regularizer(2.5e-5), activation_fn = tf.identity)

	bn = nn.BatchNorm2d(num_outputs) #size of input layer
	fc = bn(fc)
	fc = nn.ReLU(fc)
	return fc


def fc_bn_lrelu(num_inputs, num_outputs):
	fc = nn.Linear(num_inputs, num_outputs) #input feature size, output feature size
	fc.weight = nn.init.normal(fc.weight, mean=0, std=0.02)

	#not sure how to regularize the 
	#weights_regularizer=tf.contrib.layers.l2_regularizer(2.5e-5), activation_fn = tf.identity)

	bn = nn.BatchNorm2d(num_outputs) #size of input layer
	fc = bn(fc)
	fc = nn.LeakyReLU(fc)
	return fc



# To be understood more

def compute_kernel():
	pass

def compute_mmd():
	pass


#object...
class SmallLayers(nn.Module):

	def __init__(self, network):
		#super(Net, self).__init__()
		self.network = network
		#not sure
		self.relu = nn.ReLU()
		self.lrelu = nn.LeakyReLU(0.1)
		self.conv1 = nn.Conv2d(input_x, self.network.cs[1], 4, 2)

		conv = F.conv2d(inputs, weight, stride) 

		self.fc1 = nn.Linear(conv2.get_shape(), self.network.cs[3])
		nninit.constant(self.fc1.bias, 0.1)
		nninit.sigmoid(self.fc1.bias, )

		pass

	def hidden_d1(self, input_x, is_training=True):
		#MNIST; self.cs = [1, 64, 128, 1024]
		conv1 = conv2d_bn_lrelu(input_x, self.network.cs[1], [4,4], 2, is_training)
		conv2 = conv2d_bn_lrelu(conv1, self.network.cs[2], [4,4], 2, is_training)
		#conv2 = reshape(conv2, [-1, np.prod(conv2.get_shape().as_list()[1:])])
		#reshpe in PYTORCH
		conv2 = conv2.view(-1, np.prod(conv2.get_shape().as_list()[1:]))
		#fc1 = fc(conv2., self.network.cs[3], activation_fn = tf.identity)
		return fc1

	def ladder_z1(self, input_x, is_training=True):
		conv1 = conv2d_bn_lrelu(input_x, self.network.cs[1], [4,4], 2, is_training)
		conv2 = conv2d_bn_lrelu(conv1, self.netork.cs[2], [4,4], 2, is_training)
		conv2 = conv2.view(-1, np.prod(conv2.get_shape().as_list()[1:]))
		fc1_mean = fc(conv2, self.network.cs[3], activation_fn=tf.identity)
		fc1_stddev = tf.contrib.layers.fully_connected(conv2, self.network.ladder0_dim, activation_fn=tf.sigmoid)
		return fc1_mean, fc1_stddev\



	def hidden_d2():
		pass

	def ladder_z2():
		pass

	def hidden_d3():
		pass

	def ladder_z3():
		pass


	def generative_z2():
		pass

	def generative_z1():
		pass

	def generative_x():
		pass

	# To be understood
	def combine_noise():
		pass



















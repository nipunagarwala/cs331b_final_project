import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class ConfigVLAE(object):

	def __init__(self):
		self.hidden1 = ConfigHidden1()
		self.hidden2 = ConfigHidden2()
		self.hidden3 = ConfigHidden3()
		self.ladder1 = ConfigLadder1()
		self.ladder2 = ConfigLadder2()
		self.ladder3 = ConfigLadder3()
		self.generative1 = ConfigGenerative1()
		self.generative2 = ConfigGenerative2()
		self.generative3 = ConfigGenerative3()
		self.ladder1_dim = self.ladder1.linear.mean_length
		self.ladder2_dim = self.ladder2.linear3.mean_length
		self.ladder3_dim = self.ladder3.linear3.mean_length
		self.batch_size = 32


#####################################################
#####################################################
class ConfigHidden1(object):

	def __init__(self):
		self.conv1 = self.Conv1()
		self.conv2 = self.Conv2()
		self.linear = self.Linear()

	class Conv1(object):

		def __init__(self):
			self.shape = (1,28,28)
			self.num_inputs = 1
			self.num_outputs = 64
			self.kernel_size = (3,3)
			self.stride = 2
			self.padding = (1,1)
			self.dilation = 1
			self.groups = 1



	class Conv2(object):

		def __init__(self):
			self.shape = (1,28,28)
			self.num_inputs = 64
			self.num_outputs = 128
			self.kernel_size = (3,3)
			self.stride = 2
			self.padding = (1,1)
			self.dilation = 1
			self.groups = 1

	class Linear(object):

		def __init__(self):
			self.num_outputs = 64
			self.bias = False


#####################################################
#####################################################

class ConfigHidden2(object):

	def __init__(self):
		self.linear1 = self.Linear1()
		self.linear2 = self.Linear2()
		self.linear3 = self.Linear3()

	class Linear1(object):

		def __init__(self):
			self.num_inputs = 64
			self.num_outputs = 64


	class Linear2(object):

		def __init__(self):
			self.num_inputs = 64
			self.num_outputs = 64


	class Linear3(object):

		def __init__(self):
			self.num_inputs = 64
			self.num_outputs = 64


#####################################################
#####################################################


class ConfigHidden3(object):

	def __init__(self):
		self.linear1 = self.Linear1()
		self.linear2 = self.Linear2()
		self.linear3 = self.Linear3()

	class Linear1(object):

		def __init__(self):
			self.num_inputs = 64
			self.num_outputs = 64


	class Linear2(object):

		def __init__(self):
			self.num_inputs = 64
			self.num_outputs = 64


	class Linear3(object):

		def __init__(self):
			self.num_inputs = 64
			self.num_outputs = 64


#####################################################
#####################################################


class ConfigLadder1(object):

	def __init__(self):
		self.conv1 = self.Conv1()
		self.conv2 = self.Conv2()
		self.linear = self.Linear()

	class Conv1(object):

		def __init__(self):
			self.shape = (1,28,28)
			self.num_inputs = 1
			self.num_outputs = 64
			self.kernel_size = (3,3)
			self.stride = 2
			self.padding = (1,1)
			self.dilation = 1 
			self.groups = 1



	class Conv2(object):

		def __init__(self):
			self.shape = (1,28,28)
			self.num_inputs = 64
			self.num_outputs = 128
			self.kernel_size = (3,3)
			self.stride = 2
			self.padding = (1,1)
			self.dilation = 1
			self.groups = 1

	class Linear(object):

		def __init__(self):
			self.mean_length =  2
			self.stddev_length = 2


#####################################################
#####################################################


class ConfigLadder2(object):

	def __init__(self):
		self.linear1 = self.Linear1()
		self.linear2 = self.Linear2()
		self.linear3 = self.Linear3()

	class Linear1(object):

		def __init__(self):
			self.num_inputs = 64
			self.num_outputs = 64


	class Linear2(object):

		def __init__(self):
			self.num_inputs = 64
			self.num_outputs = 64


	class Linear3(object):

		def __init__(self):
			self.num_inputs = 64
			self.mean_length =  2
			self.stddev_length = 2



#####################################################
#####################################################

class ConfigLadder3(object):

	def __init__(self):
		self.linear1 = self.Linear1()
		self.linear2 = self.Linear2()
		self.linear3 = self.Linear3()

	class Linear1(object):

		def __init__(self):
			self.num_inputs = 64
			self.num_outputs = 64


	class Linear2(object):

		def __init__(self):
			self.num_inputs = 64
			self.num_outputs = 64


	class Linear3(object):

		def __init__(self):
			self.num_inputs = 64
			self.mean_length =  2
			self.stddev_length = 2



#####################################################
#####################################################

class ConfigNoise(object):

	def __init__(self):
		pass


#####################################################
#####################################################


class ConfigGenerative1(object):

	def __init__(self):
		self.conv1 = self.Conv1()
		self.conv2 = self.Conv2()
		self.linear = self.Linear()
		self.linear_init = self.LinearInit()

	class Conv1(object):

		def __init__(self):
			self.num_inputs = 3
			self.num_outputs = 64
			self.kernel_size = (3,3)
			self.stride = 2
			self.padding = (1,1)
			self.dilation = 1 
			self.groups = 1
			self.output_padding = (1,1)



	class Conv2(object):

		def __init__(self):
			self.num_inputs = 64
			self.num_outputs = 128
			self.kernel_size = (3,3)
			self.stride = 2
			self.padding = (1,1)
			self.dilation = 1
			self.groups = 1
			self.output_padding = (1,1)
			self.dim1 = 28/4
			self.dim2 = 28/4
			self.dim3 = 28/4

	class LinearInit(object):

		def __init__(self):
			self.num_inputs = 64
			self.num_outputs = 64

	class Linear(object):

		def __init__(self):
			self.num_inputs = 64
			self.num_outputs = 64



#####################################################
#####################################################


class ConfigGenerative2(object):

	def __init__(self):
		self.linear1 = self.Linear1()
		self.linear2 = self.Linear2()
		self.linear3 = self.Linear3()
		self.linear_init = self.LinearInit()

	class Linear1(object):

		def __init__(self):
			self.num_inputs = 64
			self.num_outputs = 64


	class Linear2(object):

		def __init__(self):
			self.num_inputs = 64
			self.num_outputs = 64


	class Linear3(object):

		def __init__(self):
			self.num_inputs = 64
			self.num_outputs = 64

	class LinearInit(object):

		def __init__(self):
			self.num_inputs = 2
			self.num_outputs = 64



#####################################################
#####################################################

class ConfigGenerative3(object):

	def __init__(self):
		self.linear1 = self.Linear1()
		self.linear2 = self.Linear2()
		self.linear3 = self.Linear3()

	class Linear1(object):

		def __init__(self):
			self.num_inputs = 2
			self.num_outputs = 64


	class Linear2(object):

		def __init__(self):
			self.num_inputs = 64
			self.num_outputs = 64


	class Linear3(object):

		def __init__(self):
			self.num_outputs = 64



#####################################################
#####################################################


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
		self.reg_type = 'kl'


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
			self.stride = (2,2)
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
			self.num_outputs = 1024
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
			self.num_inputs = 1024
			self.num_outputs = 1024


	class Linear2(object):

		def __init__(self):
			self.num_inputs = 1024
			self.num_outputs = 1024


	class Linear3(object):

		def __init__(self):
			self.num_inputs = 1024
			self.num_outputs = 1024


#####################################################
#####################################################


class ConfigHidden3(object):

	def __init__(self):
		self.linear1 = self.Linear1()
		self.linear2 = self.Linear2()
		self.linear3 = self.Linear3()

	class Linear1(object):

		def __init__(self):
			self.num_inputs = 1024
			self.num_outputs = 1024


	class Linear2(object):

		def __init__(self):
			self.num_inputs = 1024
			self.num_outputs = 1024


	class Linear3(object):

		def __init__(self):
			self.num_inputs = 1024
			self.num_outputs = 1024


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
			self.num_inputs = 1024
			self.num_outputs = 1024


	class Linear2(object):

		def __init__(self):
			self.num_inputs = 1024
			self.num_outputs = 1024


	class Linear3(object):

		def __init__(self):
			self.num_inputs = 1024
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
			self.num_inputs = 1024
			self.num_outputs = 1024


	class Linear2(object):

		def __init__(self):
			self.num_inputs = 1024
			self.num_outputs = 1024


	class Linear3(object):

		def __init__(self):
			self.num_inputs = 1024
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
			self.num_inputs = 64
			self.num_outputs = 1
			self.kernel_size = (3,3)
			self.stride = (2,2)
			self.padding = (1,1)
			self.dilation = 1 
			self.groups = 1
			self.output_padding = (1,1)



	class Conv2(object):

		def __init__(self):
			self.num_inputs = 128
			self.num_outputs = 64
			self.kernel_size = (3,3)
			self.stride = (2,2)
			self.padding = (1,1)
			self.dilation = 1
			self.groups = 1
			self.output_padding = (1,1)
			self.dim1 = 128
			self.dim2 = 28/4
			self.dim3 = 28/4

	class LinearInit(object):

		def __init__(self):
			self.num_inputs = 2
			self.num_outputs = 1024

	class Linear(object):

		def __init__(self):
			self.num_inputs = 1024
			self.num_outputs = 6272



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
			self.num_inputs = 1024
			self.num_outputs = 1024


	class Linear2(object):

		def __init__(self):
			self.num_inputs = 1024
			self.num_outputs = 1024


	class Linear3(object):

		def __init__(self):
			self.num_inputs = 1024
			self.num_outputs = 1024

	class LinearInit(object):

		def __init__(self):
			self.num_inputs = 2
			self.num_outputs = 1024



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
			self.num_outputs = 1024


	class Linear2(object):

		def __init__(self):
			self.num_inputs = 1024
			self.num_outputs = 1024


	class Linear3(object):

		def __init__(self):
			self.num_outputs = 1024



#####################################################
#####################################################




class ConfigVAE(object):

	def __init__(self):
		self.encoder = Encoder()
		self.decoder = Decoder()
		self.conv1 = self.encoder.Conv1()
		self.conv2 = self.encoder.Conv2()
		self.fc1 = self.encoder.FC1()
		self.fc2 = self.encoder.FC2()
		self.final = self.encoder.FCFinal()
		self.gen_fc1 = self.decoder.FC1Gen()
		self.gen_fc2 = self.decoder.FC2Gen()
		self.gen_conv1 = self.decoder.Conv1Gen()
		self.gen_conv2 = self.decoder.Conv2Gen()
		self.batch_size = 32
		self.training = True



class Encoder(object):

	def __init__(self):
		pass



	class Conv1(object):
		def __init__(self):
			self.num_inputs = 3
			self.num_outputs = 64
			self.kernel_size = (3,3)
			self.stride = (2,2)
			self.padding = (1,1)
			self.dilation = 1
			self.groups = 1
			self.shape = (3, 32, 32)



	class Conv2(object):
		def __init__(self):
			self.num_inputs = 64
			self.num_outputs = 128
			self.kernel_size = (3,3)
			self.stride = 2
			self.padding = (1,1)
			self.dilation = 1
			self.groups = 1


	class FC1(object):
		def __init__(self):
			self.num_inputs = None
			self.num_outputs = 256
			self.bias = False



	class FC2(object):
		def __init__(self):
			self.num_inputs = 256
			self.num_outputs = 256
			self.bias = False



	class FCFinal(object):
		def __init__(self):
			self.mean_len = 10
			self.stddev_len = 10


class Decoder(object):

	def __init__(self):
		pass

	class Conv1Gen(object):
		def __init__(self):
			self.num_inputs = 128
			self.num_outputs = 64
			self.kernel_size = (3,3)
			self.stride = (2,2)
			self.padding = (1,1)
			self.dilation = 1
			self.groups = 1
			self.output_padding = (1,1)


	class Conv2Gen(object):
		def __init__(self):
			self.num_inputs = 64
			self.num_outputs = 3
			self.kernel_size = (3,3)
			self.stride = (2,2)
			self.padding = (1,1)
			self.dilation = 1
			self.groups = 1
			self.output_padding = (1,1)



	class FC1Gen(object):
		def __init__(self):
			self.num_inputs = 10
			self.num_outputs = 256
			self.bias = False



	class FC2Gen(object):
		def __init__(self):
			self.num_inputs =  256
			# self.num_outputs = 6272
			self.num_outputs = 8192
			self.bias = False

import torch.nn as nn
import torch.nn.functional as F
import numpy as np



#####################################################
#####################################################
class ConfigHidden1(object):

	def __init__(self):
		pass

	class Conv1(object):

		def __init__(self):
			self.shape = (3,40,40)
			self.num_inputs = 3
			self.num_outputs = 64
			self.kernel_size = [4,4]
			self.stride = 2
			self.padding = 
			self.dilation = 1
			self.groups = 



	class Conv2(object):

		def __init__(self):
			self.num_inputs = 64
			self.num_outputs = 128
			self.kernel_size = [4,4]
			self.stride = 2
			self.padding = 
			self.dilation = 1
			self.groups = 

	class Linear(object):

		def __init__(self):
			self.num_outputs = 1024
			self.bias = False


#####################################################
#####################################################

class ConfigHidden2(object):

	def __init__(self):
		pass

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


# class ConfigHidden3(object):

# 	def __init__(self):
# 		pass

# 	class Linear1(object):

# 		def __init__(self):
# 			self.num_inputs = 
# 			self.num_outputs = 


# 	class Linear2(object):

# 		def __init__(self):
# 			self.num_inputs = 
# 			self.num_outputs = 


# 	class Linear3(object):

# 		def __init__(self):
# 			self.num_inputs = 
# 			self.num_outputs = 


#####################################################
#####################################################


class ConfigLadder1(object):

	def __init__(self):
		pass

	class Conv1(object):

		def __init__(self):
			self.num_inputs = 3
			self.num_outputs = 64
			self.kernel_size = [4,4]
			self.stride = 2
			self.padding = 
			self.dilation = 1 
			self.groups = 



	class Conv2(object):

		def __init__(self):
			self.num_inputs = 64
			self.num_outputs = 128
			self.kernel_size = [4,4]
			self.stride = 2
			self.padding = 
			self.dilation = 1
			self.groups = 

	class Linear(object):

		def __init__(self):
			self.mean_length =  2
			self.stddev_length = 2


#####################################################
#####################################################


class ConfigLadder2(object):

	def __init__(self):
		pass

	class Linear1(object):

		def __init__(self):
			self.num_inputs = 
			self.num_outputs = 


	class Linear2(object):

		def __init__(self):
			self.num_inputs = 
			self.num_outputs = 


	class Linear3(object):

		def __init__(self):
			self.num_inputs = 
			self.mean_length =  
			self.stddev_length = 



#####################################################
#####################################################

class ConfigLadder3(object):

	def __init__(self):
		pass

	class Linear1(object):

		def __init__(self):
			self.num_inputs = 
			self.num_outputs = 


	class Linear2(object):

		def __init__(self):
			self.num_inputs = 
			self.num_outputs = 


	class Linear3(object):

		def __init__(self):
			self.num_inputs = 
			self.mean_length =  
			self.stddev_length = 



#####################################################
#####################################################

class ConfigNoise(object):

	def __init__(self):
		pass


#####################################################
#####################################################


class ConfigGenerative1(object):

	def __init__(self):
		pass

	class Conv1(object):

		def __init__(self):
			self.num_inputs = 3
			self.num_outputs = 64
			self.kernel_size = [4,4]
			self.stride = 2
			self.padding = 
			self.dilation = 1 
			self.groups = 



	class Conv2(object):

		def __init__(self):
			self.num_inputs = 64
			self.num_outputs = 128
			self.kernel_size = [4,4]
			self.stride = 2
			self.padding = 
			self.dilation = 1
			self.groups = 

	class Linear(object):

		def __init__(self):
			self.mean_length =  2
			self.stddev_length = 2


#####################################################
#####################################################


class ConfigGenerative2(object):

	def __init__(self):
		pass

	class Linear1(object):

		def __init__(self):
			self.num_inputs = 
			self.num_outputs = 


	class Linear2(object):

		def __init__(self):
			self.num_inputs = 
			self.num_outputs = 


	class Linear3(object):

		def __init__(self):
			self.num_inputs = 
			self.mean_length =  
			self.stddev_length = 



#####################################################
#####################################################

class ConfigGenerative3(object):

	def __init__(self):
		pass

	class Linear1(object):

		def __init__(self):
			self.num_inputs = 
			self.num_outputs = 


	class Linear2(object):

		def __init__(self):
			self.num_inputs = 
			self.num_outputs = 


	class Linear3(object):

		def __init__(self):
			self.num_inputs = 
			self.mean_length =  
			self.stddev_length = 



#####################################################
#####################################################


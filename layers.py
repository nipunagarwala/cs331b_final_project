import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# # To be understood more

# def compute_kernel():
# 	pass

# def compute_mmd():
# 	pass


''' Class : Conv2D_BN_ReLU
	Usage : Container class to make using Convolution, BatchNorm and ReLU together easier

	config : The configuration files having all the parameters that we want to pass into the class

	Returns : Output after applying all the relevant layers
'''
class Conv2D_BN_ReLU(nn.Module):

	def __init__(self, config):
		self.conv = nn.Conv2d(config.num_inputs, config.num_outputs, config.kernel_size, stride=config.stride, 
 							padding=config.padding, dilation=config.dilation, groups=config.groups) 
		self.bn = nn.BatchNorm2d(config.num_outputs, affine=True)
		self.relu = nn.ReLU(inplace=False)


	def forward(self, x):

		output = self.relu(self.bn(self.conv(x)))

		return output


''' Class : Conv2D_BN_ReLU
	Usage : Container class to make using Fully Connected, BatchNorm and ReLU together easier

	config : The configuration files having all the parameters that we want to pass into the class

	Returns : Output after applying all the relevant layers
'''
class FC_BN_ReLU(nn.Module):

	def __init__(self, config):

		self.fc = nn.Linear(config.num_inputs, config.num_outputs, bias=False)
		self.bn = nn.BatchNorm1d(num_features=config.num_outputs, affine=True)
		self.relu = nn.ReLU(inplace=False)


	def forward(self, x):

		output = self.relu(self.bn(self.fc(x)))

		return output


''' Class : HiddenLayerConv
	Usage : Generally for the first Hidden layer of the Variational Ladder AutoEncoder we need to extract
			features, and this layer has the necessary Convolutional layers to do so. We definitely
			need one of this for VLAE, but may need more depending on how deep we go

	config_1 : First configuration class for the first set of Convolutions, BatchNorm and ReLU
	config_2 : Second configuration class for the second set of Convolutions, BatchNorm and ReLU

	Returns : Output after applying all the relevant layers
'''
class HiddenLayerConv(nn.Module):

	def __init__(self, config_1, config_2, layer_name):
		self.conv1 = Conv2D_BN_ReLU(config_1)
		self.conv2 = Conv2D_BN_ReLU(config_2)
		self.name = layer_name


	def forward(self, x):

		self.output = self.conv2(self.conv1(x))
		# self.output = self.output.view(-1, np.prod(np.asarray(self.output.size())[1:]))

		return self.output



''' Class : HiddenLayerFC
	Usage : Generally used after atleast one Hidden layer of the Variational Ladder AutoEncoder 
			having convolutions in it. This class uses fully connected layers instead of
			convolutions for feature extraction

	config_1 : First configuration class for the first set of Fully Connected, BatchNorm and ReLU layers
	config_2 : Second configuration class for the second set of Fully Connected, BatchNorm and ReLU layers
	config_3 : Third configuration class for the third set of Fully Connected, BatchNorm and ReLU layers

	Returns : Output after applying all the relevant layers

'''
class HiddenLayerFC(nn.Module):

	def __init__(self, config_1, config_2, config_3, layer_name):
		self.fc1 = FC_BN_ReLU(config_1)
		self.fc2 = FC_BN_ReLU(config_2)
		self.fc3 = nn.Linear(config_2.num_outputs, config_3.num_outputs, bias=False)

		self.name = layer_name


	def forward(self, x):

		self.output = self.fc3(self.fc2(self.fc1(x)))

		return self.output


''' Class : LadderLayerConv
	Usage : This layer is used alongside the Hidden layers to find the mean and standard deviation
			for the output of the previous layer (if present) or the input directly. At least one 
			is necessary in the VLAE to learn the corresponding Gaussian distribution and
			for feature extraction if multiple layers present.

	config_1 : First configuration class for the first set of Convolution, BatchNorm and ReLU layers
	config_2 : Second configuration class for the second set of Convolution, BatchNorm and ReLU layers

	Returns : Mean and Std Dev after applying relevant layers

'''
class LadderLayerConv(nn.Module):

	def __init__(self, config_1, config_2, config_3, layer_name):
		self.conv1 = Conv2D_BN_ReLU(config_1)
		self.conv2 = Conv2D_BN_ReLU(config_2)

		self.sigmoid = nn.Sigmoid()
		self.config_3 = config_3

		self.name = layer_name


	def forward(self,x):
		
		features = self.conv2(self.conv1(x))

		self.fc_mean = nn.Linear(np.prod(np.asarray(features.size())[1:]), self.config_3.mean_length)
		self.fc_stddev = nn.Linear(np.prod(np.asarray(features.size())[1:]), self.config_3.stddev_length)

		flattened_features = features.view(-1, np.prod(np.asarray(features.size())[1:]))
		self.mean = self.fc_mean(flattened_features)
		self.std_dev = self.fc_stddev(flattened_features)
		self.std_dev = self.sigmoid(self.std_dev)

		return self.mean, self.std_dev



''' Class : LadderLayerFC
	Usage : This layer is used after the LadderLayerConv layer to extract further features
			from the input data so that we an predict the mean and standard deviation of
			the same

	config_1 : First configuration class for the first set of Fully Connected, BatchNorm and ReLU layers
	config_2 : Second configuration class for the second set of Fully Connected, BatchNorm and ReLU layers
	config_2 : Third configuration class for the third set of Fully Connected, BatchNorm and ReLU layers

	Returns : Mean and Std Dev after applying relevant layers

'''

class LadderLayerFC(nn.Module):

	def __init__(self, config_1, config_2, config_3, layer_name):
		self.fc1 = FC_BN_ReLU(config_1)
		self.fc2 = FC_BN_ReLU(config_2)

		self.fc_mean = nn.Linear(config_2.num_outputs, config_3.mean_length)
		self.fc_stddev = nn.Linear(config_2.num_outputs, config_3.stddev_length)
		self.sigmoid = nn.Sigmoid()

		self.name = layer_name


	def forward(self,x):
		self.features =self.fc2(self.fc1(x))

		self.mean = self.fc_mean(self.features)
		self.std_dev = self.fc_stddev(self.features)
		self.std_dev = self.sigmoid(self.std_dev)

		return self.mean, self.std_dev


class GenerativeLayerConv(nn.Module):

	def __init__(self, config, layer_name):
		


	def forward(self,x):



class GenerativeLayerFC(nn.Module):

	def __init__(self, config_1, config_2, layer_name):
		


	def forward(self,x):
		



class CombineNoise(nn.Module):

	def __init__(self, config, layer_name):


	def forward(self, x)



















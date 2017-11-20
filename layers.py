import torch
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
		# TODO: Calculate Padding for the Input!!!!!!!
		output = self.relu(self.bn(self.conv(x)))

		return output


''' Class : Conv2DTrans_BN_ReLU
	Usage : Container class to make using Convolution Transpose, BatchNorm and ReLU together easier

	config : The configuration files having all the parameters that we want to pass into the class

	Returns : Output after applying all the relevant layers
'''
class Conv2DTrans_BN_ReLU(nn.Module):

	def __init__(self, config):
		self.conv_trans = nn.ConvTranspose2d(config.num_inputs, config.num_outputs, config.kernel_size, stride=config.stride, 
 							padding=config.padding, dilation=config.dilation, groups=config.groups) 
		self.bn = nn.BatchNorm2d(config.num_outputs, affine=True)
		self.relu = nn.ReLU(inplace=False)


	def forward(self, x):
		# TODO: Calculate Padding for the Input!!!!!!!
		output = self.relu(self.bn(self.conv_trans(x)))

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
	config_3 : Third configuration class for the third set of Fully Connected layers

	Returns : Output after applying all the relevant layers
'''
class HiddenLayerConv(nn.Module):

	def __init__(self, config_1, config_2, config_3, layer_name):
		self.conv1 = Conv2D_BN_ReLU(config_1)
		self.conv2 = Conv2D_BN_ReLU(config_2)

		n_size = self._get_conv_output(config_1.shape)
		self.fc =  nn.Linear(n_size, self.config_3.num_outputs, bias=self.config_3.bias)
		self.name = layer_name
		


	def _get_conv_output(self, shape):
        bs = 1
        input_dat = Variable(torch.rand(bs, *shape))
        output_feat = self._forward_features(input_dat)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

    def _forward_features(self, x):
        x = self.conv2(self.conv1(x))
        return x

	def forward(self, x):

		self.output = self.conv2(self.conv1(x))

		self.features = self.output.view(self.output.size(0), -1)
		self.output = self.fc(self.features)

		return self.output



''' Class : HiddenLayerFC
	Usage : Generally used after atleast one Hidden layer of the Variational Ladder AutoEncoder 
			having convolutions in it. This class uses fully connected layers instead of
			convolutions for feature extraction

	config_1 : First configuration class for the first set of Fully Connected, BatchNorm and ReLU layers
	config_2 : Second configuration class for the second set of Fully Connected, BatchNorm and ReLU layers
	config_3 : Third configuration class for the third set of Fully Connected layers

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
	config_2 : Third configuration class for the Mean and Standard Deviation fully connected layers

	Returns : Mean and Std Dev after applying relevant layers

'''
class LadderLayerConv(nn.Module):

	def __init__(self, config_1, config_2, config_3, layer_name):
		self.conv1 = Conv2D_BN_ReLU(config_1)
		self.conv2 = Conv2D_BN_ReLU(config_2)

		n_size = self._get_conv_output(config_1.shape)

		self.fc_mean = nn.Linear(n_size, self.config_3.mean_length)
		self.fc_stddev = nn.Linear(n_size, self.config_3.stddev_length)

		self.sigmoid = nn.Sigmoid()

		self.name = layer_name


	def _get_conv_output(self, shape):
        bs = 1
        input_dat = Variable(torch.rand(bs, *shape))
        output_feat = self._forward_features(input_dat)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

    def _forward_features(self, x):
        x = self.conv2(self.conv1(x))
        return x


	def forward(self,x):
		
		features = self.conv2(self.conv1(x))

		flattened_features = features.view(features.size(0), -1)
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
	config_2 : Third configuration class for the Mean and Standard Deviation fully connected layers

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

	def __init__(self, config_1, config_2, config_3, config_4, config_noise, layer_name):
		self.fc_init = FC_BN_ReLU(config_1)
		self.fc1 = FC_BN_ReLU(config_2)
		self.deconv1 = Conv2DTrans_BN_ReLU(config_3)
		self.deconv2 = nn.ConvTranspose2d(config_4.num_inputs, config_4.num_outputs, config_4.kernel_size, stride=config_4.stride, 
 							padding=config_4.padding, dilation=config_4.dilation, groups=config_4.groups) 
		self.comb_noise = CombineNoise(config_noise)

		self.config_3 = config_3
		self.name = layer_name


	def forward(self, latent_in, ladder_in):
		cur_state = latent_in
		if ladder_in is not None:
			cur_state = self.fc_init(ladder_in)
			if latent_in is not None:
				cur_state = self.comb_noise(latent_in, ladder_in)
			else:
				cur_state = ladder_in

		elif not self.latent_in:
			print("Generative layer must be given an input")
			exit(1)

		self.flat_vec = self.fc1(cur_state)
		self.feature_map = self.flat_vec.view(list(self.flat_vec.size())[0], self.config_3.dim1, self.config_3.dim2, self.config_3.dim3)

		self.output = self.deconv2(self.deconv1(self.feature_map))

		return self.output


class GenerativeLayerLadderFC(nn.Module):

	def __init__(self, config_1, config_2, config_3, config_4, config_noise, layer_name):

		self.fc_init = FC_BN_ReLU(config_1)
		self.fc1 = FC_BN_ReLU(config_2)
		self.fc2 = FC_BN_ReLU(config_3)
		self.fc3 = nn.Linear(config_3.num_outputs, config_4.num_outputs, bias=False)
		self.comb_noise = CombineNoise(config_noise)

		self.name = layer_name


	def forward(self, latent_in, ladder_in):
		cur_state = latent_in
		if ladder_in is not None:
			cur_state = self.fc_init(ladder_in)
			if latent_in is not None:
				cur_state = self.comb_noise(latent_in, ladder_in)
			else:
				cur_state = ladder_in

		elif not self.latent_in:
			print("Generative layer must be given an input")
			exit(1)

		self.output = self.fc3(self.fc2(self.fc1(cur_state)))

		return self.output

class GenerativeLayerSimpleFC(nn.Module):

	def __init__(self, config_1, config_2, config_3, layer_name):
		self.fc1 = FC_BN_ReLU(config_1)
		self.fc2 = FC_BN_ReLU(config_2)
		self.fc3 = nn.Linear(config_2.num_outputs, config_3.num_outputs, bias=False)

		self.name = layer_name


	def forward(self,x):
		self.output = self.fc3(self.fc2(self.fc1(x)))

		return self.output
		


class CombineNoise(nn.Module):

	def __init__(self, config, layer_name):
		self.config = config
		self.layer_name = layer_name


	def forward(self, latent_in, ladder_in, gate=None):
		if self.config.name == 'concat':
			return torch.cat((latent_in, ladder_in), len(list(ladder_in.size()))-1 )

		else:

			if self.config.name == 'add':
				return latent_in + ladder_in
			elif self.config.name == 'gated_add':
				return latent_in + gate*ladder_in
			else:
				print("Wrong method name used for CombineNoise Class")
				exit(1)




















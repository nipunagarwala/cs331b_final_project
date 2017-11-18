import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# # To be understood more

# def compute_kernel():
# 	pass

# def compute_mmd():
# 	pass


#object...
class HiddenLayerConv(nn.Module):

	def __init__(self, config, layer_name):
		self.conv1 = nn.Conv2d(config.num_inputs, config.num_outputs, config.kernel_size, stride=config.stride, 
 							padding=config.padding, dilation=config.dilation, groups=config.groups) 
		self.bn1 = nn.BatchNorm2d(config.num_outputs, affine=True)
		self.relu1 = nn.ReLU(inplace=False)

		self.conv2 = nn.Conv2d(config.num_inputs, config.num_outputs, config.kernel_size, stride=config.stride, 
 							padding=config.padding, dilation=config.dilation, groups=config.groups) 
		self.bn2 = nn.BatchNorm2d(config.num_outputs, affine=True)
		self.relu2 = nn.ReLU(inplace=False)

		self.name = layer_name


	def forward(self, x):

		self.layer1_out = self.relu1(self.bn1(self.conv1(x)))
		self.layer2_out = self.relu2(self.bn2(self.conv2(self.layer1_out)))

		output = conv2.view(-1, np.prod(self.layer2_out.get_shape().as_list()[1:]))

		return output


class HiddenLayerFC(nn.Module):

	def __init__(self, config, layer_name):
		self.fc1 = nn.Linear(config.num_inputs, config.num_outputs, bias=False)
		self.bn1 = nn.BatchNorm1d(num_features=config.num_outputs, affine=True)
		self.relu1 = nn.ReLU(inplace=False)

		self.fc1 = nn.Linear(config.num_inputs, config.num_outputs, bias=False)
		self.bn1 = nn.BatchNorm1d(num_features=config.num_outputs, affine=True)
		self.relu1 = nn.ReLU(inplace=False)

		self.fc3 = nn.Linear(config.num_inputs, config.num_outputs, bias=False)

		self.name = layer_name


	def forward(self, x):

		self.layer1_out = self.relu1(self.bn1(self.fc1(x)))
		self.layer2_out = self.relu2(self.bn2(self.fc2(self.layer1_out)))

		output = self.fc3(self.layer2_out)

		return output


class LadderLayerConv(nn.Module):

	def __init__(self, config, layer_name):
		self.conv1 = nn.Conv2d(config.num_inputs, config.num_outputs, config.kernel_size, stride=config.stride, 
 							padding=config.padding, dilation=config.dilation, groups=config.groups) 
		self.bn1 = nn.BatchNorm2d(config.num_outputs, affine=True)
		self.relu1 = nn.ReLU(inplace=False)

		self.conv2 = nn.Conv2d(config.num_inputs, config.num_outputs, config.kernel_size, stride=config.stride, 
 							padding=config.padding, dilation=config.dilation, groups=config.groups) 
		self.bn2 = nn.BatchNorm2d(config.num_outputs, affine=True)
		self.relu2 = nn.ReLU(inplace=False)

		self.fc_mean = nn.Linear(config.num_inputs, config.num_outputs)
		self.fc_stddev = nn.Linear(config.num_inputs, config.num_outputs)
		self.sigmoid = nn.Sigmoid()

		self.name = layer_name


	def forward(self,x):
		self.layer1_out = self.relu1(self.bn1(self.conv1(x)))
		self.layer2_out = self.relu2(self.bn2(self.conv2(self.layer1_out)))

		features = conv2.view(-1, np.prod(self.layer2_out.get_shape().as_list()[1:]))
		self.mean = self.fc_mean(features)
		self.std_dev = self.fc_stddev(features)
		self.std_dev = self.sigmoid(self.std_dev)

		return self.mean, self.std_dev


class LadderLayerFC(nn.Module):

	def __init__(self, config, layer_name):
		self.conv1 = nn.Conv2d(config.num_inputs, config.num_outputs, config.kernel_size, stride=config.stride, 
 							padding=config.padding, dilation=config.dilation, groups=config.groups) 
		self.bn1 = nn.BatchNorm2d(config.num_outputs, affine=True)
		self.relu1 = nn.ReLU(inplace=False)

		self.conv2 = nn.Conv2d(config.num_inputs, config.num_outputs, config.kernel_size, stride=config.stride, 
 							padding=config.padding, dilation=config.dilation, groups=config.groups) 
		self.bn2 = nn.BatchNorm2d(config.num_outputs, affine=True)
		self.relu2 = nn.ReLU(inplace=False)

		self.fc_mean = nn.Linear(config.num_inputs, config.num_outputs)
		self.fc_stddev = nn.Linear(config.num_inputs, config.num_outputs)
		self.sigmoid = nn.Sigmoid()

		self.name = layer_name


	def forward(self,x):
		self.layer1_out = self.relu1(self.bn1(self.conv1(x)))
		self.layer2_out = self.relu2(self.bn2(self.conv2(self.layer1_out)))

		features = conv2.view(-1, np.prod(self.layer2_out.get_shape().as_list()[1:]))
		self.mean = self.fc_mean(features)
		self.std_dev = self.fc_stddev(features)
		self.std_dev = self.sigmoid(self.std_dev)

		return self.mean, self.std_dev


class GenerativeLayer(nn.Module):

	def __init__(self, config, layer_name):
		


	def forward(self,x):
		



class CombineNoise(nn.Module):

	def __init__(self, config, layer_name):


	def forward(self, x)



















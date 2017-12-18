import numpy as np
import sys
import os
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from layers import *
from torch.autograd import Variable
from quantize import *



class VLadder(nn.Module):

	def __init__(self, args, config, quant_method = None):
		super(VLadder, self).__init__()
		self.hidden_d1 = HiddenLayerConv(config.hidden1.conv1, config.hidden1.conv2,
								 config.hidden1.linear, "hidden_layer_1")
		self.hidden_d2 = HiddenLayerFC(config.hidden2.linear1, config.hidden2.linear2,
											 config.hidden2.linear3, "hidden_layer_2")
		self.hidden_d3 = HiddenLayerFC(config.hidden3.linear1, config.hidden3.linear2, 
									config.hidden3.linear3, "hidden_layer_3")

		self.latent_z1 = LadderLayerConv(config.ladder1.conv1, config.ladder1.conv2,
											 config.ladder1.linear, "ladder_layer_1")
		self.latent_z2 = LadderLayerFC(config.ladder2.linear1, config.ladder2.linear2, 
											config.ladder2.linear3, "ladder_layer_2")
		self.latent_z3 = LadderLayerFC(config.ladder3.linear1, config.ladder3.linear2,
										 config.ladder3.linear3, "ladder_layer_3")

		self.generative_g3 = GenerativeLayerSimpleFC(config.generative3.linear1,
								 config.generative3.linear2, config.generative3.linear3,
								  "generative_layer_3")
		self.generative_g2 = GenerativeLayerLadderFC(config.generative2.linear1,
								 config.generative2.linear2, config.generative2.linear3,
								  "generative_layer_2", config.generative2.linear_init)
		self.generative_g1 = GenerativeLayerConv(config.generative1.linear,
								config.generative1.conv2,config.generative1.conv1, 
								 "generative_layer_1", config.generative1.linear_init)
		self.read_out = nn.Sigmoid()

		self.config = config
		self.args = args
		self.quant_method = quant_method
		self.scale = True


	def encode(self, x):
		# Define the encoding pass on the network
		output = {}

		output['d1'] = self.hidden_d1(x)
		output['z1_mu'], output['z1_sigma'] = self.latent_z1(x)

		output['d2'] = self.hidden_d2(output['d1'])
		output['z2_mu'], output['z2_sigma'] = self.latent_z2(output['d1'])

		# output['d3'] = self.hidden_d3(output['d2'])
		output['z3_mu'], output['z3_sigma'] = self.latent_z3(output['d2'])

		return output

	def decode(self, x):
		output = {}

		output['train3'] =  self.generative_g3(x['z3'])
		output['train2'] = self.generative_g2(output['train3'], x['z2'])
		output['train1'] = self.generative_g1(output['train2'], x['z1'])
		output['out'] = self.read_out(output['train1'])

		return output

	def reparametrization(self, x):
		sample = {}
		# Define the samples that we take on the mean and standard deviations
		normal_dist_var = Variable(torch.randn(self.config.batch_size, self.config.ladder1_dim).cuda(), requires_grad=False)
		sample['z1'] = x['z1_mu'] + torch.mul(x['z1_sigma'], normal_dist_var)

		normal_dist_var = Variable(torch.randn(self.config.batch_size, self.config.ladder2_dim).cuda(), requires_grad=False)
		sample['z2'] = x['z2_mu'] + torch.mul(x['z2_sigma'], normal_dist_var)

		normal_dist_var = Variable(torch.randn(self.config.batch_size, self.config.ladder3_dim).cuda(), requires_grad=False)
		sample['z3'] = x['z3_mu'] + torch.mul(x['z3_sigma'], normal_dist_var)

		return sample

	def sample(self):
		sample = {}

		sample['z1'] = Variable(torch.randn(self.config.batch_size, self.config.ladder1_dim).cuda(), requires_grad=False)
		sample['z2'] = Variable(torch.randn(self.config.batch_size, self.config.ladder2_dim).cuda(), requires_grad=False)
		sample['z3'] = Variable(torch.randn(self.config.batch_size, self.config.ladder3_dim).cuda(), requires_grad=False)

		return sample


	def generate_samples(self):
		sample = self.sample()
		output = self.decode(sample)

		return output


	def forward(self, x):
		encoding = {}
		sample = {}
		output = {}

		encoding = self.encode(x)
		sample = self.reparametrization(encoding)
		output = self.decode(sample)
		output.update(sample)
		output.update(encoding)

		return output



	def loss_function(self, output, target_y):
		reg = {}
		regularization = 0
		if self.config.reg_type == 'kl':

				reg['z1'] = -0.5*torch.sum(1 +  output['z1_sigma'] - output['z1_mu'].pow(2) - \
									       output['z1_sigma'].exp())
				reg['z2'] =  -0.5*torch.sum(1 +  output['z2_sigma'] - output['z2_mu'].pow(2) - \
									       output['z2_sigma'].exp())
				reg['z3'] =  -0.5*torch.sum(1 +  output['z3_sigma'] - output['z3_mu'].pow(2) - \
									       output['z3_sigma'].exp())

		elif self.config.reg_type == 'mmd':
			reg['z1'] = 0
			reg['z2'] = 0
			reg['z3'] = 0

		else:
			print("The regularization type is either not implemented or incorrect ...")
			exit(1)

		regularization += reg['z1'] + reg['z2'] + reg['z3']

		loss = F.binary_cross_entropy(output['out'].cuda(), target_y.cuda(), size_average=True).cuda()

		return regularization + 8*loss



class SimpleVAE(nn.Module):

	def __init__(self, config, model_name, quant_method = None, bits = None):
		super(SimpleVAE, self).__init__()

		self.conv1 = Conv2D_BN_ReLU(config.conv1)
		self.conv2 = Conv2D_BN_ReLU(config.conv2)

		n_size = self._get_conv_output(config.conv1.shape)
		config.fc1.num_inputs = n_size
		self.fc1 = FC_BN_ReLU(config.fc1)
		self.fc2 = FC_BN_ReLU(config.fc2)

		self.mean_fc = nn.Linear(config.fc2.num_outputs, config.final.mean_len, bias=False)
		self.stddev_fc = nn.Linear(config.fc2.num_outputs, config.final.stddev_len, bias=False)

		self.gen_fc1 = FC_BN_ReLU(config.gen_fc1)
		self.gen_fc2 = FC_BN_ReLU(config.gen_fc2)

		self.gen_conv1 = Conv2DTrans_BN_ReLU(config.gen_conv1)
		self.gen_conv2 = nn.ConvTranspose2d(config.gen_conv2.num_inputs, config.gen_conv2.num_outputs,
							config.gen_conv2.kernel_size, stride=config.gen_conv2.stride,
 							padding=config.gen_conv2.padding, output_padding=config.gen_conv2.output_padding,
 							dilation=config.gen_conv2.dilation, groups=config.gen_conv2.groups)
		self.read_out = nn.Sigmoid()

		self.config = config
		self.name = model_name
		self.quant_method = quant_method
		self.bits = bits
		self.scale = 0
		self.sf = None



	def _get_conv_output(self, shape):
		bs = 1
		input_dat = Variable(torch.rand(bs, *shape))
		output_feat = self._forward_features(input_dat)
		n_size = output_feat.data.view(bs, -1).size(1)

		return n_size

	def _forward_features(self, x):
		x = self.conv1(x)
		x = self.conv2(x)
		return x

	def encoder(self, x):
		conv_out = self.conv2(self.conv1(x))

		features = conv_out.view(conv_out.size(0), -1)
		mid = self.fc2(self.fc1(features))

		mean_vec = self.mean_fc(mid)
		log_stddev = self.stddev_fc(mid)
		self.conv_out = conv_out

		return mean_vec, log_stddev


	def decoder(self, x):

		mid_gen = self.gen_fc2(self.gen_fc1(x))

		mid_gen = mid_gen.view(mid_gen.size(0), self.conv_out.size(1),
								self.conv_out.size(2), self.conv_out.size(3))

		output = self.gen_conv1(mid_gen)
		output = self.read_out(self.gen_conv2(output))
		del self.conv_out

		return output

	def reparametrization(self, mean_vec, log_stddev):
		if self.training:
			# normal_dist_var = Variable(torch.randn(self.config.batch_size, self.config.final.stddev_len).cuda().half(), requires_grad=False)
			normal_dist_var = Variable(torch.randn(self.config.batch_size, self.config.final.stddev_len).cuda(), requires_grad=False)
			# std = self.log_stddev.mul(0.5).exp_()
			# eps = Variable(std.data.new(std.size()).normal_())
			sample = mean_vec + torch.mul(log_stddev, normal_dist_var)
			# self.sample = self.mean_vec + torch.mul(self.log_stddev.exp(), normal_dist_var)
			return sample
		else:
			return mean_vec



	def forward(self, x):
		np.set_printoptions(threshold='nan')
		finalIn = x
		if self.quant_method != None:
			if self.scale < 3:
				sf_new = self.bits - 1 - compute_integral_part(finalIn, 0)
				self.sf = min(self.sf, sf_new) if self.sf is not None else sf_new
				self.scale += 1
			elif self.quant_method == 'linear':
				# print("OH YEAH")
				# self.scale += 1
				finalIn = linear_quantize(x, self.sf, self.bits)
			elif self.quant_method == 'log':
				finalIn = log_linear_quantize(x, self.sf, self.bits)

		# if self.scale == 4:
		# 	print np.where(x.data.cpu().numpy()[0] > 0.0)
		# 	print np.where(finalIn.data.cpu().numpy()[0] > 0.0)
		# 	exit(1)
		
		output = {}

		mean_vec, log_stddev = self.encoder(finalIn)
		sample = self.reparametrization(mean_vec, log_stddev)
		decoder_out = self.decoder(sample)
		output['mu'] = mean_vec
		output['sigma'] = log_stddev
		output['out'] = decoder_out

		return output


	def loss_function(self, output, target_y):

		reg = -0.5*torch.sum(1 + output['sigma'] - \
							output['mu'].pow(2) - output['sigma'].exp())
		# reg = torch.mean(reg)
		regularization = 0
		regularization += reg

		# print pred_y
		# print target_y
		pred_out = output['out'].clamp(min=1e-4, max=1)
		target_y = target_y.clamp(min=1e-4, max=1)
		pred_out[pred_out != pred_out] = 0
		loss = F.binary_cross_entropy(pred_out, target_y, size_average=True).cuda()
		# self.loss = nn.MSELoss(size_average=False)

		return regularization + loss






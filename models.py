import numpy as np
import sys
import os
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from layers import *
from torch.autograd import Variable



class VLadder(nn.Module):

	def __init__(self, args, config):
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


	def encode(self, x):
		pass

	def decode(self, x):
		pass

	def reparametrization(self, x):
		pass




	def forward(self, x):
		# Define the encoding pass on the network
		output = {}
		sample = {}
		output['d1'] = self.hidden_d1(x)
		output['z1_mu'], output['z1_sigma'] = self.latent_z1(x)

		output['d2'] = self.hidden_d2(output['d1'])
		output['z2_mu'], output['z2_sigma'] = self.latent_z2(output['d1'])

		output['d3'] = self.hidden_d3(output['d2'])
		output['z3_mu'], output['z3_sigma'] = self.latent_z3(output['d2'])


		# Define the samples that we take on the mean and standard deviations
		if self.args.phase == 'train':
			normal_dist_var = Variable(torch.randn(self.config.batch_size, self.config.ladder1_dim).cuda(), requires_grad=False)
			sample['z1'] = output['z1_mu'] + torch.mul(output['z1_sigma'], normal_dist_var)

			normal_dist_var = Variable(torch.randn(self.config.batch_size, self.config.ladder1_dim).cuda(), requires_grad=False)
			sample['z2'] = output['z2_mu'] + torch.mul(output['z2_sigma'], normal_dist_var)

			normal_dist_var = Variable(torch.randn(self.config.batch_size, self.config.ladder1_dim).cuda(), requires_grad=False)
			sample['z3'] = output['z3_mu'] + torch.mul(output['z3_sigma'], normal_dist_var)


		# Defines the generative and the decoder network
		if self.args.phase == 'train':
			output['train3'] =  self.generative_g3(sample['z3'])
			output['train2'] = self.generative_g2(output['train3'], sample['z2'])
			output['train1'] = self.generative_g1(output['train2'], sample['z1'])
			output['out'] = self.read_out(output['train1'])

		elif self.args.phase == 'generate':

			normal_dist_var1 = Variable(torch.randn(self.config.batch_size, self.config.ladder1_dim), requires_grad=False)
			normal_dist_var2 = Variable(torch.randn(self.config.batch_size, self.config.ladder2_dim), requires_grad=False)
			normal_dist_var3 = Variable(torch.randn(self.config.batch_size, self.config.ladder3_dim), requires_grad=False)

			output['gen3'] = self.generative_g3(normal_dist_var3)
			output['gen2'] = self.generative_g2(output['gen3'], normal_dist_var2)
			output['gen1'] = self.generative_g1(output['gen2'], normal_dist_var1)
			output['out'] = self.read_out(output['gen1'])

		else:
			print("Option has either not been implemented or is incorrect ......") 
			exit(1)


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

		return regularization


	#####################################################
	#####################################################
	# TODO : Implement this functionality
		# def generate_op():
		# 	pass

		# def generate_conditional_samples_op():
		# 	pass

		# # To be understood
		# def generate_manifold_samples_op():
		# 	pass


	#####################################################
	#####################################################






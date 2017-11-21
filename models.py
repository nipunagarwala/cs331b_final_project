import numpy as np
import sys
import os
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from layers import *



class VLadder(nn.Module):

	def __init__(self, args, config):
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

		self.generative_g3 = GenerativeLayerFC(config.generative3.linear1,
								 config.generative3.linear2, config.generative3.linear3,
								  "generative_layer_3")
		self.generative_g2 = GenerativeLayerFC(config.generative2.linear1,
								 config.generative2.linear2, config.generative2.linear3,
								  "generative_layer_2")
		self.generative_g1 = GenerativeLayerConv(config.generative1.conv1,
								 config.generative1.conv2, config.generative1.linear, 
								 "generative_layer_1")

		self.output = {}
		self.reg = {}
		self.config = config
		self.args = args
		self.regularization = 0


	def forward(self, x_in, in_gen_3, in_gen_2, in_gen_1):
		# Define the encoding pass on the network
		self.output['d1'] = self.hidden_d1(x)
		self.output['z1_mu'], self.output['z1_sigma'] = self.latent_z1(x)

		self.output['d2'] = self.hidden_d2(self.output['d1'])
		self.output['z2_mu'], self.output['z2_sigma'] = self.latent_z2(self.output['d1'])

		self.output['d3'] = self.hidden_d3(self.output['d2'])
		self.output['z3_mu'], self.output['z3_sigma'] = self.latent_z2(self.output['d2'])


		# Define the samples that we take on the mean and standard deviations
		if self.args.phase == 'train':
			normal_dist_var = Variable(torch.randn(self.config.batch_size, self.config.ladder_dim['z1']), requires_grad=False)
			self.sample['z1'] = self.output['z1_mu'] + torch.mul(self.output['z1_sigma'], normal_dist_var)

			normal_dist_var = Variable(torch.randn(self.config.batch_size, self.config.ladder_dim['z2']), requires_grad=False)
			self.sample['z2'] = self.output['z2_mu'] + torch.mul(self.output['z2_sigma'], normal_dist_var)

			normal_dist_var = Variable(torch.randn(self.config.batch_size, self.config.ladder_dim['z3']), requires_grad=False)
			self.sample['z3'] = self.output['z3_mu'] + torch.mul(self.output['z3_sigma'], normal_dist_var)


		# Defines the generative and the decoder network
		if self.args.phase == 'train':

			self.output['train3'] =  self.generative_g3(None, self.sample['z3'])
			self.output['train2'] = self.generative_g3(self.output['train3'], self.sample['z2'])
			self.output['train1'] = self.generative_g3(self.output['train2'], self.sample['z1'])
			self.output['out'] = self.output['train1']

		elif self.args.phase == 'generate':

			self.output['gen3'] = self.generative_g3(None, in_gen_3)
			self.output['gen2'] = self.generative_g3(self.output['gen3'], in_gen_2)
			self.output['gen1'] = self.generative_g3(self.output['gen2'], in_gen_1)
			self.output['out'] = self.output['gen1']

		else:
			print("Option has either not been implemented or is incorrect ......") 
			exit(1)


		return self.output['out']



	def regularization_function(self):

		if self.config.reg_type == 'kl':

				self.reg['z1'] = 0.5*torch.sum(torch.exp(self.self.output['z1_mu']) + \
									 torch.mul(self.output['z1_mu'],self.output['z1_mu']) - 1 - self.output['z1_sigma'])
				self.reg['z2'] = 0.5*torch.sum(torch.exp(self.self.output['z2_mu']) + \
									 torch.mul(self.output['z2_mu'],self.output['z2_mu']) - 1 - self.output['z2_sigma'])
				self.reg['z3'] = 0.5*torch.sum(torch.exp(self.self.output['z3_mu']) + \
									 torch.mul(self.output['z3_mu'],self.output['z3_mu']) - 1 - self.output['z3_sigma'])

		elif self.config.reg_type == 'mmd':

			self.reg['z1'] = 0
			self.reg['z2'] = 0
			self.reg['z3'] = 0

		else
			print("The regularization type is either not implemented or incorrect ...")
			exit(1)

		self.regularization += self.reg['z1'] + self.reg['z2'] + self.reg['z3']

		return self.regularization


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






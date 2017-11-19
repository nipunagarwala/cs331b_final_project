import numpy as np
import sys
import os
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from layers import *



class VLadder(object):

	def __init__(self, cmdline_args, config):
		self.config = config
		if self.config.net_type == 'small':
			self.layers = SmallLayers()
		else:
			print("The network type is either not valid or not implement.")
			exit(1)



	# Creates the Encoder network of the VLAE
	def create_encoder_network():
		self.means = {}
		self.std_dev = {}
		self.hidden_layers = {}
		self.ladder_sample = {}
		self.ladder_reg = {}

		if self.config.num_layers > 0:
			self.mean['z1'], self.std_dev['z1'] = self.layers.ladder_z1()
			normal_dist_var = Variable(torch.randn(self.config.batch_size, self.config.ladder_dim['z1']), requires_grad=False)
			self.ladder_sample['z1'] = self.mean['z1'] + torch.mul(self.std_dev['z1'],normal_dist_var)

			if self.config.reg_type == 'kl':
				self.ladder_reg['z1'] = 0.5*torch.sum(torch.exp(self.std_dev['z1']) + \
									 torch.mul(self.mean['z1'],self.mean['z1']) - 1 - self.std_dev['z1'])
			elif self.config.reg_type == 'mmd':
				self.ladder_reg['z1'] = 0
			else
				print("The regularization type is either not implemented or incorrect ...")

			self.regularization += self.ladder_reg['z1']

		for i in range(2, self.config.num_layers+1):

			if i == 2:
				cur_hidden_func = self.layers.hidden_d1
				cur_ladder_func = self.layers.ladder_z2
			elif i == 3:
				cur_hidden_func = self.layers.hidden_d2
				cur_ladder_func = self.layers.ladder_z3
			elif i == 4:
				cur_hidden_func = self.layers.hidden_d3
				cur_ladder_func = self.layers.ladder_z4
			else:
				print("The current iteration loop for layers has not been implemented or is incorrect")
				exit(1)

			self.hidden_layers['d'+str(i-1)] = cur_hidden_func()
			self.mean['z'+str(i)], self.std_dev['z'+str(i)] = cur_ladder_func(self.hidden_layers['d'+str(i-1)], self.config)

			normal_dist_var = Variable(torch.randn(self.config.batch_size, self.config.ladder_dim['z'+str(i)]), requires_grad=False)
			self.ladder_sample['z'+str(i)] = self.mean['z'+str(i)] + torch.mul(self.std_dev['z'+str(i)],normal_dist_var)

			if self.config.reg_type == 'kl':
				self.ladder_reg['z'+str(i)] = 0.5*torch.sum(torch.exp(self.std_dev['z'+str(i)]) + \
									 torch.mul(self.mean['z'+str(i)],self.mean['z'+str(i)]) - 1 - self.std_dev['z'+str(i)])
			elif self.config.reg_type == 'mmd':
				self.ladder_reg['z'+str(i)] = 0
			else
				print("The regularization type is either not implemented or incorrect ...")

			self.regularization += self.ladder_reg['z'+str(i)]



	# Creates the generative network of the VLAE
	def create_decoder_network(gen_data_list):
		self.ladder_rep = {}

		gen_data_list_rev = reversed(gen_data_list)

		for i in range(self.config.num_layers, 0, -1):

			if i == 1:
				cur_gen_func = self.layers.generative_x
			elif i == 2:
				cur_gen_func = self.layers.generative_z1
			elif i == 3:
				cur_gen_func = self.layers.generative_z2
			elif i == 4:
				cur_gen_func = self.layers.generative_z3
			else:
				print("The current iteration loop for layers has not been implemented or is incorrect")
				exit(1)

			self.ladder_gen_in['z'+str(i)] = Variable(gen_data_list_rev[i-1])
			self.ladder_rep['z'+str(i)] = [ladder_cur_noise , self.config.ladder_dim['z'+str(i)], 
											self.ladder_sample['z'+str(i)] ]

			if i < self.config.num_layers:
				self.train_latent_st['zhat'+str(i)] = cur_gen_func(self.train_latent_st['zhat'+str(i+1)],
													self.ladder_sample['z'+str(i)], self.config)
				self.gen_latent_st['zhat'+str(i)] = cur_gen_func(self.gen_latent_st['zhat'+str(i+1)], 
													self.ladder_gen_in['z'+str(i)], self.config)
			else:
				self.train_latent_st['zhat'+str(i)] = cur_gen_func(None, self.ladder_sample['z'+str(i)], self.config)
				self.gen_latent_st['zhat'+str(i)] = cur_gen_func(None, self.ladder_gen_in['z'+str(i)], self.config)

		self.train_out = self.train_latent_st['zhat1']
		self.gen_out = self.gen_latent_st['zhat1']

	def loss_op():
		 self.reconst_loss = tf.reduce_mean(tf.abs(self.toutput - self.target_placeholder))

	def train_op():
		pass

	def test_op():
		pass

	def inference_op():
		pass

	def generate_op():
		pass

	def generate_conditional_samples_op():
		pass

	# To be understood
	def generate_manifold_samples_op():
		pass



class VLadder(nn.Module):

	def __init__(self, config):
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
								 config.generative3.linear2, config.generative3.linear3, "generative_layer_3")
		self.generative_g2 = GenerativeLayerFC(config.generative2.linear1,
								 config.generative2.linear2, config.generative2.linear3, "generative_layer_2")
		self.generative_g1 = GenerativeLayerConv(config.generative1.conv1,
								 config.generative1.conv2, config.generative1.linear, "generative_layer_1")



	def forward(self, x):








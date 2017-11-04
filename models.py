import numpy as np
import sys
import os
from layers import SmallLayers
import torch



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

		if self.config.num_layers > 0:
			self.ladder_z1_mean, self.ladder_z1_std_dev = self.layers.ladder_z1()
			normal_dist_var = torch.Variable(torch.randn(self.config.batch_size, self.config.ladder_z1_dim), requires_grad=False)
			self.ladder_z1_sample = self.ladder_z1_mean + torch.mul(self.ladder_z1_std_dev,normal_dist_var)

			if self.config.reg_type == 'kl':
				self.ladder_z1_reg = 0.5*(torch.exp(self.ladder_z1_std_dev) + \
									 torch.mul(self.ladder_z1_mean,self.ladder_z1_mean) - 1 - self.ladder_z1_std_dev)
			elif self.config.reg_type == 'mmd':
				self.ladder_z1_reg = 
			else
				print("The regularization type is either not implemented or incorrect ...")

			self.regularization += self.ladder_z1_reg


		if self.config.num_layers > 1:
			self.hidden_layer_1 = self.layers.hidden_d1()
			self.ladder_z2_mean, self.ladder_z2_std_dev = self.layers.ladder_z2(self.hidden_layer_1)

			normal_dist_var = torch.Variable(torch.randn(self.config.batch_size, self.config.ladder_z2_dim), requires_grad=False)
			self.ladder_z2_sample = self.ladder_z2_mean + torch.mul(self.ladder_z2_std_dev,normal_dist_var)

			if self.config.reg_type == 'kl':
				self.ladder_z2_reg = 0.5*(torch.exp(self.ladder_z2_std_dev) + \
									 torch.mul(self.ladder_z2_mean,self.ladder_z2_mean) - 1 - self.ladder_z2_std_dev)
			elif self.config.reg_type == 'mmd':
				self.ladder_z2_reg = 
			else
				print("The regularization type is either not implemented or incorrect ...")

			self.regularization += self.ladder_z2_reg


		if self.config.num_layers > 2:
			self.hidden_layer_2 = self.layers.hidden_d2()
			self.ladder_z3_mean, self.ladder_z3_std_dev = self.layers.ladder_z3(self.hidden_layer_2)

			normal_dist_var = torch.Variable(torch.randn(self.config.batch_size, self.config.ladder_z3_dim), requires_grad=False)
			self.ladder_z3_sample = self.ladder_z3_mean + torch.mul(self.ladder_z3_std_dev,normal_dist_var)

			if self.config.reg_type == 'kl':
				self.ladder_z3_reg = 0.5*(torch.exp(self.ladder_z3_std_dev) + \
									 torch.mul(self.ladder_z3_mean,self.ladder_z3_mean) - 1 - self.ladder_z3_std_dev)
			elif self.config.reg_type == 'mmd':
				self.ladder_z3_reg = 
			else
				print("The regularization type is either not implemented or incorrect ...")

			self.regularization += self.ladder_z3_reg


		if self.config.num_layers > 3:
			self.hidden_layer_3 = self.layers.hidden_d3()
			self.ladder_z4_mean, self.ladder_z4_std_dev = self.layers.ladder_z4(self.hidden_layer_3)

			normal_dist_var = torch.Variable(torch.randn(self.config.batch_size, self.config.ladder_z4_dim), requires_grad=False)
			self.ladder_z4_sample = self.ladder_z4_mean + torch.mul(self.ladder_z4_std_dev,normal_dist_var)

			if self.config.reg_type == 'kl':
				self.ladder_z4_reg = 0.5*(torch.exp(self.ladder_z4_std_dev) + \
									 torch.mul(self.ladder_z4_mean,self.ladder_z4_mean) - 1 - self.ladder_z4_std_dev)
			elif self.config.reg_type == 'mmd':
				self.ladder_z4_reg = 
			else
				print("The regularization type is either not implemented or incorrect ...")

			self.regularization += self.ladder_z4_reg



	# Creates the generative network of the VLAE
	def create_decoder_network():
		self.ladder_representations = {}

		if self.config.num_layers > 3:
			


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




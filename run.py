from utils import parseCommandLine
import numpy as np
import os
import sys

class Config(object):

	def __init__(self, config):
		self.input_shape = 
		self.target_shape = 
		self.is_training = 
		self.reg_val = 
		self.reg_type = 
		self.num_layers = 
		self.dataset_path = 
		self.ladder_z1_dim = 
		self.ladder_z2_dim = 
		self.ladder_z3_dim = 
		self.batch_size = 
		self.latent_noise = 
		self.dilation = 
		self.stride = 
		self.reshape = True



def train_model(args):
	cur_config = Config()
	curModel = VLadder(args, cur_config)




def main(args):
	if args.phase == 'train':
		train_model(args)
	else:
		print("The phase option parse is either incorrect or not implemented yet. Sorry for the inconvenience .....")
		exit(1)


if '__name__' == '__main__':
	args = parseCommandLine()
	main(args)

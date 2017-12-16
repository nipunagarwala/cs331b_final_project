import numpy as np 
import os
import sys
from argparse import ArgumentParser
from multiprocessing import Pool
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt



def parseCommandLine():
	desc = u'{0} [Args] [Options]\nDetailed options -h or --help'.format(__file__)
	parser = ArgumentParser(description=desc)

	print("Parsing Command Line Arguments...")
	requiredTrain = parser.add_argument_group('Required Train/Test arguments')
	requiredTrain.add_argument('-p', choices = ["train", "test", 'generate'], type = str,
						dest = 'phase', required = True, help = 'Training or Testing phase to be run')
	requiredTrain.add_argument('-m', choices = ["vlae"], type = str,
						dest = 'model', required = True, help = 'Model under consideration')

	parser.add_argument('-ckpt', dest='ckpt_dir', default='/diskhdd/CS331b/checkpoint/', 
									type=str, help='Set the checkpoint directory')
	parser.add_argument('-data', dest='data_dir', default='/diskhdd/CS331b/datasets/', 
									type=str, help='Set the data directory')
	

	args = parser.parse_args()
	return args



def quantizeParseArgs():
	parser = ArgumentParser(description='PyTorch SVHN Example')
	parser.add_argument('--quant_method', default='linear', help='linear|minmax|log|tanh')
	parser.add_argument('--seed', type=int, default=117, help='random seed (default: 1)')

	# parser.add_argument('--input_size', type=int, default=224, help='input size of image')
	# parser.add_argument('--n_sample', type=int, default=20, help='number of samples to infer the scaling factor')
	parser.add_argument('--param_bits', type=int, default=8, help='bit-width for parameters')
	parser.add_argument('--bn_bits', type=int, default=32, help='bit-width for running mean and std')
	# parser.add_argument('--fwd_bits', type=int, default=8, help='bit-width for layer output')
	parser.add_argument('--overflow_rate', type=float, default=0.0, help='overflow rate')
	parser.add_argument('--phase', choices = ['quantize', 'tune'], type = str,
						dest = 'phase', required = True, help = 'Simple Quantization of Model or Fine tuning of quantized model`')
	args = parser.parse_args()

	return args

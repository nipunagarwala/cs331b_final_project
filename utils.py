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
	requiredTrain.add_argument('-p', choices = ["train", "test", "dev"], type = str,
						dest = 'phase', required = True, help = 'Training or Testing phase to be run')
	requiredTrain.add_argument('-m', choices = ["vlae_small"], type = str,
						dest = 'model', required = True, help = 'Model under consideration')

	parser.add_argument('-ckpt', dest='ckpt_dir', default='/diskhdd/CS331b/checkpoint/', 
									type=str, help='Set the checkpoint directory')
	parser.add_argument('-data', dest='data_dir', default='/diskhdd/CS331b/datasets/', 
									type=str, help='Set the data directory')
	

	args = parser.parse_args()
	return args

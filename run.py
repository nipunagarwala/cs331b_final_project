from utils import parseCommandLine
import numpy as np
import os
import sys


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

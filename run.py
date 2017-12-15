from utils import parseCommandLine
import numpy as np
import os
import sys
from config import *
from models import *
from torch import optim
from torchvision import datasets, transforms
import torch

dtype = torch.cuda.FloatTensor
torch.cuda.manual_seed(1)
kwargs = {'num_workers': 1, 'pin_memory': True}

def run_test_epoch(epoch, model, optimizer, test_loader):
	model.eval()
	test_loss = 0
	for i, (data, _) in enumerate(test_loader):
		# if args.cuda:
		data = Variable(data, volatile=True)
		data = data.cuda()
		output_dict = model(data)
		loss = model.loss_function(output_dict, data).data[0]
		# if i == 0:
		# 	n = min(data.size(0), 8)
		# 	comparison = torch.cat([data[:n],
		# 					recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
		# 	save_image(comparison.data.cpu(),
		# 				'/diskhdd/cs333/results_' + str(epoch) + '.png', nrow=n)

	test_loss /= len(test_loader.dataset)
	print('====> Test set loss: {:.4f}'.format(test_loss))



def run_train_epoch(epoch, model, optimizer, train_loader, args, batch_size):
	model.train()
	train_loss = 0
	for batch_idx, (data, _) in enumerate(train_loader):
		if(data.size(0) != batch_size):
			break
		data = Variable(data)
		data = data.cuda()
		optimizer.zero_grad()


		
		output_dict = model(data)
		loss = model.loss_function(output_dict, data)
		loss.backward(retain_graph=True)
		train_loss += loss.data[0]
		optimizer.step()
		# if batch_idx % args.log_interval == 0:
		# if batch_idx  == 0:
		print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
			epoch, batch_idx * len(data), len(train_loader.dataset),
			100. * batch_idx / len(train_loader),
			loss.data[0] / len(data)))

	print('====> Epoch: {} Average loss: {:.4f}'.format(
		  epoch, train_loss / len(train_loader.dataset)))


def train_model(args):
	num_epochs = 50
	batch_size = 32

	cur_config = ConfigVLAE()
	curModel = VLadder(args, cur_config)
	curModel.cuda()
	optimizer = optim.Adam(curModel.parameters(), lr=1e-3)

	train_loader = torch.utils.data.DataLoader(
			 datasets.MNIST('/diskhdd/cs331b/data', train=True, download=True,
		          transform=transforms.ToTensor()),
					 batch_size=batch_size, shuffle=True, **kwargs)
	test_loader = torch.utils.data.DataLoader(
					datasets.MNIST('/diskhdd/cs331b/data', train=False, transform=transforms.ToTensor()),
					batch_size=batch_size, shuffle=True, **kwargs)

	for epoch in range(1, num_epochs + 1):
		run_train_epoch(epoch, curModel, optimizer, train_loader, args, batch_size)
		torch.save(curModel.state_dict(), '/diskhdd/cs331b/checkpoints/training_ckpt-' + str(epoch) + '.pt')
		run_test_epoch(epoch, curModel, optimizer, test_loader)




def main(args):
	if args.phase == 'train':
		train_model(args)
	else:
		print("The phase option parse is either incorrect or not implemented yet. Sorry for the inconvenience .....")
		exit(1)


if __name__ == "__main__":
	args = parseCommandLine()
	main(args)

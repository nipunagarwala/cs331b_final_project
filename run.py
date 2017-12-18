from utils import parseCommandLine
import numpy as np
import os
import sys
from config import *
from models import *
from torch import optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
import torch
import matplotlib.pyplot as plt

dtype = torch.cuda.FloatTensor
torch.cuda.manual_seed(1)
kwargs = {'num_workers': 1, 'pin_memory': True}


def run_generation(num_samples, model, batch_size):
	model.train()
	output_dict = model.generate_samples()
	output_dict = model.generate_samples()
	output_dict = model.generate_samples()
	model.eval()
	output_dict = model.generate_samples()
	comparison = output_dict['out'].view(batch_size, 1, 28, 28)
	save_image(comparison.data.cpu(),
						'/diskhdd/cs331b/results/generation_results_new' + '.png', nrow=8)


def run_test_epoch(epoch, model, optimizer, test_loader, batch_size):
	model.eval()
	test_loss = 0
	last_val = 0
	for i, (data, _) in enumerate(test_loader):
		# if args.cuda:
		if(data.size(0) != batch_size):
			last_val = data.size(0)
			break
		data = Variable(data, volatile=True)
		# data = data.cuda().half()
		data = data.cuda()
		output_dict = model(data)
		test_loss += model.loss_function(output_dict, data).data[0]
		if i == 0:
			n = min(data.size(0), 8)
			# comparison = torch.cat([data[:n],
			# 				output_dict['out'].view(batch_size, 1, 28, 28)[:n]])
			comparison = torch.cat([data[:n],
							output_dict['out'].view(batch_size, 3, 32, 32)[:n]])
			save_image(comparison.data.cpu().type(torch.cuda.FloatTensor),
						'/diskhdd/cs331b/results/svhn_results/results_' + str(epoch) + '.png', nrow=n)

	test_loss /= (len(test_loader.dataset) - last_val)
	print('====> Test set loss: {:.4f}'.format(test_loss))
	return test_loss



def run_train_epoch(epoch, model, referenceModel, optimizer, train_loader, args, batch_size):
	model.train()
	train_loss = 0
	for batch_idx, (data, _) in enumerate(train_loader):
		if(data.size(0) != batch_size):
			break
		data = Variable(data)
		# data = data.cuda().half()
		data = data.cuda()
		# model.zero_grad()
		optimizer.zero_grad()
		output_dict = model(data)
		loss = model.loss_function(output_dict, data)
		# loss = loss*32
		loss.backward()
		train_loss += loss.data[0]

		# for param, shared_param in zip(model.parameters(), referenceModel.parameters()):
		# 	shared_param.grad = param.grad.type(torch.cuda.FloatTensor)/32
		# 	shared_param.grad.data.clamp(-1000, 1000)

		optimizer.step()
		# model.load_state_dict(referenceModel.state_dict())

		if batch_idx % 32*10 == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epoch, batch_idx * len(data), len(train_loader.dataset),
				100. * batch_idx / len(train_loader),
				loss.data[0] / len(data)))

	print('====> Epoch: {} Average loss: {:.4f}'.format(
		  epoch, train_loss / len(train_loader.dataset)))
	return train_loss / len(train_loader.dataset)/16


def model_generate_samples(args):
	batch_size = 8

	cur_config = ConfigVLAE()
	curModel = VLadder(args, cur_config)
	curModel.load_state_dict(torch.load('/diskhdd/cs331b/checkpoints/new_training_ckpt-11.pt'))
	curModel.cuda()
	run_generation(batch_size, curModel, batch_size)


def train_model(args):
	num_epochs = 10
	batch_size = 32

	if args.model == 'vlae':
		cur_config = ConfigVLAE()
		curModel = VLadder(args, cur_config)
	elif args.model == 'vae':
		cur_config = ConfigVAE()
		curModel = SimpleVAE(cur_config, 'SimpleVAE')
		referenceModel = SimpleVAE(cur_config, 'SimpleVAE')
	else:
		print("Wrong model has been input ....")
		exit(1)

	curModel.cuda()
	# curModel.cuda().half()
	# referenceModel.cuda()

	optimizer = optim.Adam(curModel.parameters(), lr=1e-3)

	# train_loader = torch.utils.data.DataLoader(
	# 		 datasets.MNIST('/diskhdd/cs331b/data', train=True, download=True,
	# 	          transform=transforms.ToTensor()),
	# 				 batch_size=batch_size, shuffle=True, **kwargs)
	# test_loader = torch.utils.data.DataLoader(
	# 				datasets.MNIST('/diskhdd/cs331b/data', train=False, transform=transforms.ToTensor()),
	# 				batch_size=batch_size, shuffle=True, **kwargs)

	train_loader = torch.utils.data.DataLoader(
			 datasets.SVHN('/diskhdd/cs331b/svhn', split='train', download=True,
		          transform=transforms.ToTensor()),
					 batch_size=batch_size, shuffle=True, **kwargs)
	test_loader = torch.utils.data.DataLoader(
					datasets.SVHN('/diskhdd/cs331b/svhn', split='test', download=True, 
					transform=transforms.ToTensor()),
					batch_size=batch_size, shuffle=True, **kwargs)



	train_loss_list = []
	test_loss_list = []
	for epoch in range(1, num_epochs + 1):
		train_loss = run_train_epoch(epoch, curModel,referenceModel, optimizer, train_loader, args, batch_size)
		torch.save(curModel.state_dict(), '/diskhdd/cs331b/checkpoints/svhn_ckpt/training_ckpt-' + str(epoch) + '.pt')
		test_loss = run_test_epoch(epoch, curModel, optimizer, test_loader, batch_size)
		# train_loss_list.append(train_loss)
		# test_loss_list.append(test_loss)

	# f = plt.figure()
	# plt.plot(train_loss_list)
	# plt.plot(test_loss_list)
	# plt.xlabel('Epochs')
	# plt.ylabel('Average Loss')
	# plt.title('16-bit Floating Point Training')
	# plt.legend(['Train', 'Test'], loc='upper right')
	# plt.savefig('low_precision_loss.png', bbox_inches='tight')
	# plt.close(f)


def plot_mnist_histogram(args):
	batch_size = 32
	train_loader = torch.utils.data.DataLoader(
			 datasets.MNIST('/diskhdd/cs331b/data', train=True, download=True,
		          transform=transforms.ToTensor()),
					 batch_size=batch_size, shuffle=True, **kwargs)
	test_loader = torch.utils.data.DataLoader(
					datasets.MNIST('/diskhdd/cs331b/data', train=False, transform=transforms.ToTensor()),
					batch_size=batch_size, shuffle=True, **kwargs)

	all_data = []

	for batch_idx, (data, _) in enumerate(train_loader):
		all_data.append(data.numpy())

	print all_data[0].shape
	all_data = np.concatenate(all_data, axis=0)
	f = plt.figure()
	weights = np.ones_like(all_data.flatten())/float(len(all_data.flatten()))
	plt.hist(all_data.flatten(), bins=15, weights=weights)
	plt.xlabel('Numerical Values of Input Data')
	plt.ylabel('Frequency')
	plt.title('Distribution of MNIST values')
	plt.savefig('mnist_input_hist.png', bbox_inches='tight')
	plt.close(f)

def plot_vae_hist(args):
	cur_config = ConfigVAE()
	curModel = SimpleVAE(cur_config, 'SimpleVAE')

	all_val = []
	for p in curModel.parameters():
		all_val.append(p.data.numpy().flatten())

	all_val = np.concatenate(all_val, axis=0)
	f = plt.figure()
	weights = np.ones_like(all_val.flatten())/float(len(all_val.flatten()))
	plt.hist(all_val.flatten(), bins=20, weights=weights)
	plt.xlabel('Numerical Values of Weights')
	plt.ylabel('Probability')
	plt.title('Distribution of VAE Weight values')
	plt.savefig('vae_weight_hist.png', bbox_inches='tight')
	plt.close(f)


def main(args):
	if args.phase == 'train':
		train_model(args)
	elif args.phase == 'generate':
		model_generate_samples(args)

	else:
		print("The phase option parse is either incorrect or not implemented yet. Sorry for the inconvenience .....")
		exit(1)


if __name__ == "__main__":
	args = parseCommandLine()
	main(args)
	# plot_vae_hist(args)
	# plot_mnist_histogram(args)

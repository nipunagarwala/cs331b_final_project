import argparse
from quantize import *
import torch
from collections import OrderedDict
from config import *
from models import *
import torch
from utils import *
from torchvision import datasets, transforms
from torchvision.utils import save_image
import torch

torch.cuda.manual_seed(1)
kwargs = {'num_workers': 1, 'pin_memory': True}


def quantize_model(model, args):
	print("Starting Quantization of the model with {0} bits".format(args.param_bits))

	if args.param_bits < 32:
		state_dict = model.state_dict()
		state_dict_quant = OrderedDict()
		sf_dict = OrderedDict()
		for k, v in state_dict.items():
			if 'running' in k:
				if args.bn_bits >=32:
					print("Ignoring {}".format(k))
					state_dict_quant[k] = v
					continue
				else:
					bits = args.bn_bits
			else:
				bits = args.param_bits

			if args.quant_method == 'linear':
				sf = bits - 1. - compute_integral_part(v, overflow_rate=args.overflow_rate)
				v_quant  = linear_quantize(v, sf, bits=bits)
			elif args.quant_method == 'log':
				v_quant = log_minmax_quantize(v, bits=bits)
			elif args.quant_method == 'minmax':
				v_quant = min_max_quantize(v, bits=bits)
			else:
				v_quant = tanh_quantize(v, bits=bits)
			state_dict_quant[k] = v_quant
			# print(k, bits)

		model.load_state_dict(state_dict_quant)

		print("Completed Quantization of the Model ...")
		return model
	else:
		print("Nothing to quantize ....")
		exit(1)


def run_test_epoch(epoch, model, test_loader, batch_size):
	model.eval()
	test_loss = 0
	last_val = 0
	for i, (data, _) in enumerate(test_loader):
		# if args.cuda:
		if(data.size(0) != batch_size):
			last_val = data.size(0)
			break
		data = Variable(data, volatile=True)
		data = data.cuda()
		output_dict = model(data)
		test_loss += model.loss_function(output_dict, data).data[0]
		if i == 0:
			n = min(data.size(0), 8)
			comparison = torch.cat([data[:n],
							output_dict['out'].view(batch_size, 1, 28, 28)[:n]])
			save_image(comparison.data.cpu(),
						'/diskhdd/cs331b/results/quant_results/results_' + str(epoch) + '.png', nrow=n)

	test_loss /= (len(test_loader.dataset) - last_val)
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
		loss.backward()
		train_loss += loss.data[0]
		optimizer.step()
		if batch_idx % 32*10 == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epoch, batch_idx * len(data), len(train_loader.dataset),
				100. * batch_idx / len(train_loader),
				loss.data[0] / len(data)))

	print('====> Epoch: {} Average loss: {:.4f}'.format(
		  epoch, train_loss / len(train_loader.dataset)))


def tune_model(args):
	num_epochs = 50
	batch_size = 32

	cur_config = ConfigVLAE()
	curModel = VLadder(args, cur_config)
	curModel.cuda()
	optimizer = optim.Adam(curModel.parameters(), lr=2e-4)

	train_loader = torch.utils.data.DataLoader(
			 datasets.MNIST('/diskhdd/cs331b/data', train=True, download=True,
		          transform=transforms.ToTensor()),
					 batch_size=batch_size, shuffle=True, **kwargs)
	test_loader = torch.utils.data.DataLoader(
					datasets.MNIST('/diskhdd/cs331b/data', train=False, transform=transforms.ToTensor()),
					batch_size=batch_size, shuffle=True, **kwargs)

	for epoch in range(1, num_epochs + 1):
		run_train_epoch(epoch, curModel, optimizer, train_loader, args, batch_size)
		torch.save(curModel.state_dict(), '/diskhdd/cs331b/checkpoints/new_training_ckpt-' + str(epoch) + '.pt')
		run_test_epoch(epoch, curModel, optimizer, test_loader, batch_size)

def load_datasets(batch_size):
	train_loader = torch.utils.data.DataLoader(
			 datasets.MNIST('/diskhdd/cs331b/data', train=True, download=True,
		          transform=transforms.ToTensor()),
					 batch_size=batch_size, shuffle=True, **kwargs)
	test_loader = torch.utils.data.DataLoader(
					datasets.MNIST('/diskhdd/cs331b/data', train=False, transform=transforms.ToTensor()),
					batch_size=batch_size, shuffle=True, **kwargs)
	return train_loader, test_loader


def quantize_and_test(args):
	batch_size = 32
	cur_config = ConfigVLAE()
	curModel = VLadder(args, cur_config)
	curModel.load_state_dict(torch.load('/diskhdd/cs331b/checkpoints/normal_ckpt/new_training_ckpt-11.pt'))
	curModel.cuda()
	quantModel = quantize_model(curModel, args)
	train_loader, test_loader = load_datasets(batch_size)
	run_test_epoch('quant', quantModel, train_loader, batch_size)
	torch.save(quantModel.state_dict(), '/diskhdd/cs331b/checkpoints/quant_ckpt/quant_ckpt_test.pt')



def main(args):
	if args.phase == 'quantize':
		quantize_and_test(args)
	elif args.phase == 'tune':
		tune_model(args)



if __name__ == '__main__':
	args = quantizeParseArgs()
	main(args)





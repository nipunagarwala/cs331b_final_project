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
from torch import optim
import matplotlib
import matplotlib.pyplot as plt

torch.cuda.manual_seed(1)
kwargs = {'num_workers': 1, 'pin_memory': True}


def quantize_model(model, args):
	# print("Starting Quantization of the model with {0} bits".format(args.param_bits))

	if args.param_bits < 32:
		state_dict = model.state_dict()
		state_dict_quant = OrderedDict()
		sf_dict = OrderedDict()
		for k, v in state_dict.items():
			if 'running' in k:
				if args.bn_bits >=32:
					# print("Ignoring {}".format(k))
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

		# print("Completed Quantization of the Model ...")
		return model
	else:
		print("Nothing to quantize ....")
		exit(1)


def run_test_epoch(epoch, model, test_loader, batch_size, path):
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
			n = min(data.size(0), 16)
			comparison = torch.cat([data[:n],
							output_dict['out'].view(batch_size, 1, 28, 28)[:n]])
			save_image(comparison.data.cpu(),
						path, nrow=8)

	test_loss /= (len(test_loader.dataset) - last_val)
	print('====> Test set loss: {:.4f}'.format(test_loss))
	return test_loss



def run_train_epoch(epoch, model, referenceModel, optimizer, train_loader, args, batch_size):
	model.train()
	train_loss = 0
	model = quantize_model(model, args)
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

		for param, shared_param in zip(model.parameters(), referenceModel.parameters()):
			shared_param.grad = param.grad

		# referenceModel.load_state_dict(ref_state_dict)
		optimizer.step()
		model.load_state_dict(referenceModel.state_dict())
		quantize_model(model, args)

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

	if args.model == 'vlae':
		cur_config = ConfigVLAE()
		curModel = VLadder(args, cur_config)
		referenceModel = VLadder(args, cur_config)
	elif args.model == 'vae':
		cur_config = ConfigVAE()
		curModel = SimpleVAE(cur_config, 'SimpleVAE')
		curModel.load_state_dict(torch.load('/diskhdd/cs331b/checkpoints/simple_vae_ckpt/training_ckpt-6.pt'))
		referenceModel = SimpleVAE(cur_config, 'SimpleVAE')
		referenceModel.load_state_dict(torch.load('/diskhdd/cs331b/checkpoints/simple_vae_ckpt/training_ckpt-6.pt'))

	curModel.cuda()
	referenceModel.cuda()
	optimizer = optim.Adam(referenceModel.parameters(), lr=2e-4)

	train_loader, test_loader = load_datasets(batch_size)

	for epoch in range(1, num_epochs + 1):
		run_train_epoch(epoch, curModel, referenceModel, optimizer, train_loader, args, batch_size)
		torch.save(curModel.state_dict(), '/diskhdd/cs331b/checkpoints/simple_vae_ckpt/quant_ckpt_stoch_tuned_4-' + str(epoch) + '.pt')
		run_test_epoch(epoch, curModel, test_loader, batch_size)

def load_datasets(batch_size):
	train_loader = torch.utils.data.DataLoader(
			 datasets.MNIST('/diskhdd/cs331b/data', train=True, download=True,
				  transform=transforms.ToTensor()),
					 batch_size=batch_size, shuffle=True, **kwargs)
	test_loader = torch.utils.data.DataLoader(
					datasets.MNIST('/diskhdd/cs331b/data', train=False, transform=transforms.ToTensor()),
					batch_size=batch_size, shuffle=True, **kwargs)
	return train_loader, test_loader


def quantize_and_test(args, path):
	batch_size = 32
	if args.model == 'vlae':
		cur_config = ConfigVLAE()
		curModel = VLadder(args, cur_config)
	elif args.model == 'vae':
		cur_config = ConfigVAE()
		curModel = SimpleVAE(cur_config, 'SimpleVAE')
		curModel.load_state_dict(torch.load('/diskhdd/cs331b/checkpoints/simple_vae_ckpt/training_ckpt-6.pt'))

	curModel.cuda()
	quantModel = quantize_model(curModel, args)
	train_loader, test_loader = load_datasets(batch_size)
	cur_loss = run_test_epoch('quant', quantModel, test_loader, batch_size, path)
	return cur_loss
	# torch.save(quantModel.state_dict(), '/diskhdd/cs331b/checkpoints/quant_ckpt/quant_ckpt_test.pt')

def act_quantize_test(args):
	batch_size = 32
	if args.model == 'vlae':
		cur_config = ConfigVLAE()
		curModel = VLadder(args, cur_config)
	elif args.model == 'vae':
		cur_config = ConfigVAE()
		curModel = SimpleVAE(cur_config, 'SimpleVAE')
		curModel.load_state_dict(torch.load('/diskhdd/cs331b/checkpoints/simple_vae_ckpt/training_ckpt-6.pt'))

	print("HOWWWDYYYY")
	curModel.cuda()
	quantModel = quantize_model(curModel, args)

	train_loader, test_loader = load_datasets(batch_size)
	run_test_epoch('quant', act_quant_model, test_loader, batch_size)


# def perceptial_loss(ideal_recon, low_prec_recon):

def collect_quant_graphs(args):
	num_prec = [16, 8, 4, 3, 2]
	quant_types = ['linear', 'log', 'minmax']
	image_path = '/diskhdd/cs331b/results/simple_vae/'
	loss_arr = np.zeros((len(quant_types), len(num_prec)))

	for i in range(len(quant_types)):
		args.quant_method = quant_types[i]
		for j in range(len(num_prec)):
			if quant_types[i] == 'log' and num_prec[j] == 2:
				continue
			args.param_bits = num_prec[j]
			final_path = image_path + 'quant_test_prec_' + str(num_prec[j]) + '_method_' + quant_types[i] + '.png'
			loss_arr[i,j] = quantize_and_test(args, final_path)



	for i in range(len(quant_types)):
		f = plt.figure()
		fileName = 'loss_plot_wt_' + quant_types[i] + '.png'
		if quant_types[i] == 'log':
			plt.plot(num_prec[:4], loss_arr[i,:4])
		else:
			plt.plot(num_prec, loss_arr[i])

		plt.xlabel('Weights Integer Precision')
		plt.ylabel('Reconstruction Loss')
		plt.savefig(fileName, bbox_inches='tight')
		plt.close(f)




def main(args):
	if args.phase == 'quantize':
		quantize_and_test(args)
	elif args.phase == 'tune':
		tune_model(args)
	elif args.phase == 'act_quantize':
		act_quantize_test(args)



if __name__ == '__main__':
	args = quantizeParseArgs()
	# main(args)
	collect_quant_graphs(args)





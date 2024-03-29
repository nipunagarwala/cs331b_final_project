from torch.autograd import Variable
import torch
from torch import nn
from collections import OrderedDict
import math
import numpy as np
import random



def prob_round(x, prec = 0):
    fixup = np.sign(x) * 10**prec
    x *= fixup
    is_up = np.random.random(x.shape) < (x-x.astype(int))
    # is_up_float = is_up.astype(int)
    # is_down_float = np.logical_not(is_up.astype(int))
    # round_up_half = np.ceil(is_up_float*x)
    # round_down_half = np.floor(is_down_float*x)
    round_func = math.ceil if is_up else math.floor
    # return (round_up_half + round_down_half)/fixup
    return round_func(x) / fixup


def compute_integral_part(input, overflow_rate):
	abs_value = input.abs().view(-1)
	sorted_value = abs_value.sort(dim=0, descending=True)[0]
	split_idx = int(overflow_rate * len(sorted_value))
	v = sorted_value[split_idx]
	if isinstance(v, Variable):
		v = v.data.cpu().numpy()[0]

	sf = math.ceil(math.log(v+1e-12, 2))
	return sf

def linear_quantize(input, sf, bits):
	assert bits >= 1, bits
	if bits == 1:
		return torch.sign(input) - 1
	delta = math.pow(2.0, -sf)
	bound = math.pow(2.0, bits-1)
	min_val = - bound
	max_val = bound - 1
	rounded = torch.floor(input / delta + 0.5)
	# rounded = prob_round(input.cpu().numpy() / delta + 0.5)
	# rounded = torch.from_numpy(rounded).cuda()
	clipped_value = torch.clamp(rounded, min_val, max_val) * delta
	return clipped_value

def log_minmax_quantize(input, bits):
	assert bits >= 1, bits
	if bits == 1:
		return torch.sign(input), 0.0, 0.0

	s = torch.sign(input)
	input0 = torch.log(torch.abs(input) + 1e-20)
	v = min_max_quantize(input0, bits)
	v = torch.exp(v) * s
	return v

def log_linear_quantize(input, sf, bits):
	assert bits >= 1, bits
	if bits == 1:
		return torch.sign(input), 0.0, 0.0

	s = torch.sign(input)
	input0 = torch.log(torch.abs(input) + 1e-20)
	v = linear_quantize(input0, sf, bits)
	v = torch.exp(v) * s
	return v

def min_max_quantize(input, bits):
	assert bits >= 1, bits
	if bits == 1:
		return torch.sign(input) - 1
	min_val, max_val = input.min(), input.max()

	if isinstance(min_val, Variable):
		max_val = float(max_val.data.cpu().numpy()[0])
		min_val = float(min_val.data.cpu().numpy()[0])

	max_val += 1e-3
	input_rescale = (input - min_val) / (max_val - min_val)

	n = math.pow(2.0, bits) - 1
	v = torch.floor(input_rescale * n + 0.5) / n

	v =  v * (max_val - min_val) + min_val
	return v

def tanh_quantize(input, bits):
	assert bits >= 1, bits
	if bits == 1:
		return torch.sign(input)
	input = torch.tanh(input) # [-1, 1]
	input_rescale = (input + 1.0) / 2 #[0, 1]
	n = math.pow(2.0, bits) - 1
	v = torch.floor(input_rescale * n + 0.5) / n
	v = 2 * v - 1 # [-1, 1]

	v = 0.5 * torch.log((1 + v) / (1 - v)) # arctanh
	return v


class LinearQuant(nn.Module):
	def __init__(self, name, bits, sf=None, overflow_rate=0.0, counter=10):
		super(LinearQuant, self).__init__()
		self.name = name
		self._counter = counter

		self.bits = bits
		self.sf = sf
		self.overflow_rate = overflow_rate

	@property
	def counter(self):
		return self._counter

	def forward(self, input):
		if self._counter > 0:
			self._counter -= 1
			sf_new = self.bits - 1 - compute_integral_part(input, self.overflow_rate)
			self.sf = min(self.sf, sf_new) if self.sf is not None else sf_new
			return input
		else:
			output = linear_quantize(input, self.sf, self.bits)
			return output

	def __repr__(self):
		return '{}(sf={}, bits={}, overflow_rate={:.3f}, counter={})'.format(
			self.__class__.__name__, self.sf, self.bits, self.overflow_rate, self.counter)

		

class LogQuant(nn.Module):
	def __init__(self, name, bits, sf=None, overflow_rate=0.0, counter=10):
		super(LogQuant, self).__init__()
		self.name = name
		self._counter = counter

		self.bits = bits
		self.sf = sf
		self.overflow_rate = overflow_rate

	@property
	def counter(self):
		return self._counter

	def forward(self, input):
		if self._counter > 0:
			self._counter -= 1
			log_abs_input = torch.log(torch.abs(input))
			sf_new = self.bits - 1 - compute_integral_part(log_abs_input, self.overflow_rate)
			self.sf = min(self.sf, sf_new) if self.sf is not None else sf_new
			return input
		else:
			output = log_linear_quantize(input, self.sf, self.bits)
			return output

	def __repr__(self):
		return '{}(sf={}, bits={}, overflow_rate={:.3f}, counter={})'.format(
			self.__class__.__name__, self.sf, self.bits, self.overflow_rate, self.counter)

class NormalQuant(nn.Module):
	def __init__(self, name, bits, quant_func):
		super(NormalQuant, self).__init__()
		self.name = name
		self.bits = bits
		self.quant_func = quant_func

	@property
	def counter(self):
		return self._counter

	def forward(self, input):
		output = self.quant_func(input, self.bits)
		return output

	def __repr__(self):
		return '{}(bits={})'.format(self.__class__.__name__, self.bits)

def duplicate_model_with_quant(model, bits, overflow_rate=0.0, counter=10, type='linear'):
	"""assume that original model has at least a nn.Sequential"""
	# assert type in ['linear', 'minmax', 'log', 'tanh']
	# if isinstance(model, nn.Conv2d) or isinstance(model, nn.Linear) or isinstance(model, nn.BatchNorm2d) \
	# 		or isinstance(model, nn.BatchNorm1d) or isinstance(model, nn.ConvTranspose2d):
	# 	l = OrderedDict()
	# 	# for k, v in model._modules.items():
	# 	print("IM INNNNNN")
	# 	if isinstance(model, (nn.Conv2d, nn.Linear, nn.BatchNorm1d, nn.BatchNorm2d, nn.ConvTranspose2d, nn.AvgPool2d)):
	# 		l[k] = model
	# 		if type == 'linear':
	# 			quant_layer = LinearQuant('{}_quant'.format(k), bits=bits, overflow_rate=overflow_rate, counter=counter)
	# 		elif type == 'log':
	# 			# quant_layer = LogQuant('{}_quant'.format(k), bits=bits, overflow_rate=overflow_rate, counter=counter)
	# 			quant_layer = NormalQuant('{}_quant'.format(k), bits=bits, quant_func=log_minmax_quantize)
	# 		elif type == 'minmax':
	# 			quant_layer = NormalQuant('{}_quant'.format(k), bits=bits, quant_func=min_max_quantize)
	# 		else:
	# 			quant_layer = NormalQuant('{}_quant'.format(k), bits=bits, quant_func=tanh_quantize)
	# 		l['{}_{}_quant'.format(k, type)] = quant_layer
	# 	else:
	# 		l[k] = duplicate_model_with_quant(model, bits, overflow_rate, counter, type)
	# 	m = nn.Sequential(l)
	# 	return m
	# else:
	# 	print("HOWDDDYYYYY")
	# 	for k, v in model._modules.items():
	# 		model._modules[k] = duplicate_model_with_quant(v, bits, overflow_rate, counter, type)
	# 	return model

	l = OrderedDict()
	# print model._modules.items()
	# exit(1)
	# print model.state_dict()
	for k, v in model._modules.items():
		if isinstance(v, (nn.Conv2d, nn.Linear, nn.BatchNorm1d, nn.BatchNorm2d, nn.ConvTranspose2d, nn.ReLU, nn.Sigmoid)):
			l[k] = v
			print k
			if type == 'linear':
				quant_layer = LinearQuant('{}_quant'.format(k), bits=bits, overflow_rate=overflow_rate, counter=counter)
			elif type == 'log':
				# quant_layer = LogQuant('{}_quant'.format(k), bits=bits, overflow_rate=overflow_rate, counter=counter)
				quant_layer = NormalQuant('{}_quant'.format(k), bits=bits, quant_func=log_minmax_quantize)
			elif type == 'minmax':
				quant_layer = NormalQuant('{}_quant'.format(k), bits=bits, quant_func=min_max_quantize)
			else:
				quant_layer = NormalQuant('{}_quant'.format(k), bits=bits, quant_func=tanh_quantize)
			l['{}_{}_quant'.format(k, type)] = quant_layer
		else:
			for nextK, nextV in v._modules.items():
				# print nextK
				l[nextK] = nextV
				if type == 'linear':
					quant_layer = LinearQuant('{}_quant'.format(nextK), bits=bits, overflow_rate=overflow_rate, counter=counter)
				elif type == 'log':
					# quant_layer = LogQuant('{}_quant'.format(k), bits=bits, overflow_rate=overflow_rate, counter=counter)
					quant_layer = NormalQuant('{}_quant'.format(nextK), bits=bits, quant_func=log_minmax_quantize)
				elif type == 'minmax':
					quant_layer = NormalQuant('{}_quant'.format(nextK), bits=bits, quant_func=min_max_quantize)
				else:
					quant_layer = NormalQuant('{}_quant'.format(nextK), bits=bits, quant_func=tanh_quantize)
				l['{}_{}_{}_quant'.format(v, nextK, type)] = quant_layer

	m = nn.Sequential(l)
	return m


	


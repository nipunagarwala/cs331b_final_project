import argparse
from quantize
import torch
from collections import OrderedDict




def quantize_model(model):

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
				sf = bits - 1. - quant.compute_integral_part(v, overflow_rate=args.overflow_rate)
				v_quant  = quant.linear_quantize(v, sf, bits=bits)
			elif args.quant_method == 'log':
				v_quant = quant.log_minmax_quantize(v, bits=bits)
			elif args.quant_method == 'minmax':
				v_quant = quant.min_max_quantize(v, bits=bits)
			else:
				v_quant = quant.tanh_quantize(v, bits=bits)
			state_dict_quant[k] = v_quant
			print(k, bits)

		model.load_state_dict(state_dict_quant)



if __name__ == '__main__':
	args = quantizeParseArgs()
	




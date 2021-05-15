import torch.nn as nn
import torch

class HCEncoder_Atten(nn.Module):
	def __init__(self, nc=1):
		super(HCEncoder_Atten, self).__init__()

		ks = [3, 3, 3, 3, 3, 3, 3, 3, 3]
		ps = [1, 1, 1, 1, 1, 1, 1, 1, 1] 
		ss = [1, 1, 1, 1, 1, 1, 1, 1, 1]
		nm = [32, 32, 32, 64, 64, 64, 128, 128, 128]

		cnn = nn.Sequential()

		def convRelu(i, batchNormalization=False):
			nIn = nc if i == 0 else nm[i - 1]
			nOut = nm[i]
			cnn.add_module('conv{0}'.format(i),
						   nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
			cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
			cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

		convRelu(0)
		convRelu(1)
		convRelu(2)
		cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 32*100*20
		convRelu(3)
		convRelu(4)
		convRelu(5)
		cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 64*50*10
		convRelu(6)
		convRelu(7)
		convRelu(8)
		# cnn.add_module('pooling{0}'.format(2), nn.MaxPool2d(2, 2))  # 128*25*5

		self.cnn = cnn


	def forward(self, input):
		conv_out = self.cnn(input)

		return conv_out

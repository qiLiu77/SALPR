import torch.nn as nn
import torch

class FCDecoder(nn.Module):
	def __init__(self, nclass, input_dim=512, K =8):
		super(FCDecoder, self).__init__()
		self.input_dim = input_dim
		self.nclass = nclass
		self.fc = nn.Linear(self.input_dim, self.nclass)

	def forward(self, input):
		preds = self.fc(input)

		return preds
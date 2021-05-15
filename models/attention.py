import torch.nn as nn
import torch
import torch.nn.functional as F

class Attention_module_FC(nn.Module):

    def __init__(self, nc, K=8, downsample=4):
        # Attention_module(nc=128, K=8)
        super(Attention_module_FC, self).__init__()
        self.K = K

        nm = [512,256,128]
        atten_0 = nn.Sequential()
        # print(nm[1],nm[2])
        atten_0.add_module('conv_a_0',nn.Conv2d(nc, nm[1], 3, 1, 1))
        atten_0.add_module('bn_a_0', nn.BatchNorm2d(nm[1]))
        atten_0.add_module('relu_a_0', nn.ReLU(True))
        atten_0.add_module('pooling_a_0',nn.MaxPool2d((2, 2)))

        atten_1 = nn.Sequential()
        atten_1.add_module('conv_a_1',nn.Conv2d(nm[1], nm[2], 3, 1, 1))
        atten_1.add_module('bn_a_1', nn.BatchNorm2d(nm[2]))
        atten_1.add_module('relu_a_1', nn.ReLU(True))
        atten_1.add_module('pooling_a_1',nn.MaxPool2d((2, 2)))

        self.atten_0 = atten_0 #first two character branch
        self.atten_1 = atten_1 #last five-six character branch

        Fc_dimension = int(96*32/downsample/downsample/16)
        self.atten_fc1 = nn.Linear(Fc_dimension, Fc_dimension)
        self.atten_fc2 = nn.Linear(Fc_dimension, Fc_dimension)

        self.cnn_1_1 = nn.Conv2d(nm[1],64,1,1,0)

        self.relu    = nn.ReLU(inplace=True)
        self.sigmoid    = nn.Sigmoid()

        self.deconv1 = nn.ConvTranspose2d(nm[2], 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(64)
        self.deconv2 = nn.ConvTranspose2d(64, self.K, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(self.K)

    def forward(self, input):
        # conv features
        batch_size = input.size(0)
        conv_out = input

        x0 = self.atten_0(conv_out)

        x1 = self.atten_1(x0)

        channel = x1.size(1)
        height = x1.size(2)
        width = x1.size(3)
        fc_x = x1.view(batch_size, channel, -1)

        fc_atten = self.atten_fc2(self.atten_fc1(fc_x))
        # print(fc_atten.size())
        fc_atten = fc_atten.reshape(batch_size, channel, height, width)

        score = self.relu(self.deconv1(fc_atten))
        score = self.bn1(score+self.cnn_1_1(x0))
        atten = self.sigmoid(self.deconv2(score))

        atten_list = torch.chunk(atten, self.K, 1)
        atten = atten.reshape(batch_size, self.K, -1)
        conv_out = conv_out.reshape(conv_out.size(0), conv_out.size(1), -1)

        conv_out = conv_out.permute(0,2,1)

        atten_out = torch.bmm(atten, conv_out)
        atten_out = atten_out.view(batch_size, self.K, -1)
        
        return atten_list, atten_out
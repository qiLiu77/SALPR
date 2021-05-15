from models.encoder import *
from models.decoder import *
from models.attention import *


class LPR_model(nn.Module):

    def __init__(self, nc, nclass, imgW=96, imgH=32, K=8):
        super(LPR_model, self).__init__()

        # self.encoder = HCEncoder(nc)
        self.encoder = HCEncoder_Atten(nc)

        # self.attention = Attention_module(nc=128, K=8)
        self.attention = Attention_module_FC(nc=128, K=K, downsample=4)
        # self.attention = Attention_transformer(embedding_size=256,max_decode_len=8,encoder_channel=128)
        # self.attention = Attention_transformer_new(128,128,8,4,2,128)
        # self.attention = Attention_module_FC_ab(nc=128, K=8, downsample=4)
        # self.attention = Attention_module_FC1(nc=128, K=8, downsample=4)
        # self.attention = Attention_module_FC3(nc=128, K=8, downsample=4)

        # self.decoder = FC_noshare_Decoder(nclass, input_dim=int(imgW*imgH/8/8*128))
        self.decoder = FCDecoder(nclass, input_dim=128)
        # self.decoder = FC2Decoder(nclass, input_dim=128)


    def forward(self, input):
        conv_out = self.encoder(input)
        # print(conv_out.size())

        atten_list, atten_out = self.attention(conv_out)

        # preds = self.decoder(atten_out)
        preds = self.decoder(atten_out)
        

        # return atten_list, preds
        return atten_out, preds
        

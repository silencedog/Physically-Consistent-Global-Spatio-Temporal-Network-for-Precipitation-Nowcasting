import torch
import torch.nn as nn
from core.layers.MIMOLayers import EncoderLayer, DecoderLayer
import copy
from utils import preprocess
import sys
import torch.nn.functional as F

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Encoder(nn.Module):
    def __init__(self, channel, model_channel, N, heads):
        super().__init__()
        self.N = N
        self.model_channel = model_channel
        self.layers = get_clones(EncoderLayer(model_channel, heads), N)
        self.conv = nn.Sequential(
            # nn.Conv2d(channel, model_channel, 3, 2, 1, bias=False),  # down 2 times
            nn.Conv2d(channel, model_channel, 3, 2, 1, bias=False),   # Keep
            nn.GroupNorm(1, model_channel),
            nn.SiLU(inplace=True),
        )

    def forward(self, x, LK):
        B, T, C, H, W = x.shape   # 28 5 16 32 32
        x = self.conv(x.view(-1, *x.shape[2:]))  # 125 128 16 16
        x = x.view(B, T, self.model_channel, x.size(2), x.size(3))  # 25 5 128 16 16

        for i in range(self.N):
            x = self.layers[i](x, LK)
            x = x * LK + x
        return x


class Decoder(nn.Module):
    def __init__(self, model_channel, channel, N, heads):
        super().__init__()
        self.N = N
        self.model_channel = model_channel
        self.channel = channel
        self.layers = get_clones(DecoderLayer(self.model_channel, heads), N)
        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(self.model_channel, self.channel, 3, 1, 1, bias=False),
            nn.GroupNorm(1, self.channel),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.channel, self.channel, 3, 1, 1, bias=False),
        )
        self.conv_last = nn.Sequential(
            nn.Conv2d(2 * self.N * self.channel, 2 * self.channel, 1, 1, 0, bias=False),
            nn.GroupNorm(1, 2 * self.channel),
            nn.SiLU(inplace=True),

        )

    def forward(self, e_outputs, x, LK):
        outputs = []
        B, T, C, H, W = x.shape  # 25 5 128 16 16
        overlap = 16
        capture = 56
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, LK)
            x = x * LK + x
            x1 = self.conv(x.view(-1, *x.shape[2:]))  # 125 16 32 32
            x1 = x1.view(B, T, self.channel, x1.size(2), x1.size(3))  # 25 5 16 32 32
            outputs.append(x1)
        output = torch.cat(outputs, dim=2)

        output = self.conv_last(output.contiguous().view(output.size(0), -1, *output.shape[3:])).view(output.size(0),
                                                                                                      output.size(1),
                                                                                                      self.channel,
                                                                                                      output.size(3),
                                                                                                      output.size(4))

        return output


class Transformer(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.encoder = Encoder(opt.num_hidden, opt.d_model, opt.n_encoder_layers, opt.heads)
        self.decoder = Decoder(opt.d_model, opt.num_hidden, opt.n_decoder_layers, opt.heads)

        self.conv = nn.Sequential(
            nn.Conv2d(opt.num_hidden, opt.d_model, 3, 2, 1, bias=False),
            nn.GroupNorm(1, opt.d_model),
            nn.SiLU(inplace=True),
        )



        self.d_model = opt.d_model


    def forward(self, src, LK):

        src = src.contiguous()  # 25 5 16 32 32

        de_input = self.conv(src.view(-1, *src.shape[2:])).view(src.size(0), src.size(1), self.d_model, src.size(3)//2, src.size(4)//2)
        e_outputs = self.encoder(src, LK)  # 12 1 16 32 32
        d_output = self.decoder(e_outputs, de_input, LK)


        return d_output


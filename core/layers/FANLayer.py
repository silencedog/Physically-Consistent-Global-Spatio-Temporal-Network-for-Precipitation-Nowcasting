import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    def __init__(self, d_model, out_model):
        super().__init__()

        self.ff1 = nn.Sequential(
            nn.Conv3d(d_model, 2 * d_model, 3, 1, 1, bias=False),
            nn.GroupNorm(1, 2 * d_model),
            nn.SiLU(inplace=True),
        )
        self.ff2 = nn.Sequential(
            nn.Conv3d(2 * d_model, out_model, 3, 1, 1, bias=False),
            nn.GroupNorm(1, out_model),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        x = self.ff2(self.ff1(x.permute(0, 2, 1, 3, 4))).permute(0, 2, 1, 3, 4)
        return x

class FANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True, with_gate = True):
        super(FANLayer, self).__init__()
        # self.input_linear_p = nn.Linear(input_dim, output_dim//4, bias=bias)
        # self.input_linear_g = nn.Linear(input_dim, (output_dim-output_dim//2))
        self.ff_g = FeedForward(input_dim, (output_dim-output_dim//2))
        self.ff_p = FeedForward(input_dim, output_dim // 4)
        self.activation = nn.GELU()        
        if with_gate:
            self.gate = nn.Parameter(torch.randn(1, dtype=torch.float32))
    
    def forward(self, src):
        b, s, c, h, w = src.size()
        g = self.ff_g(src)
        g = self.activation(g)
        p = self.ff_p(src)
        
        if not hasattr(self, 'gate'):
            output = torch.cat((torch.cos(p), torch.sin(p), g), dim=2)
        else:
            gate = torch.sigmoid(self.gate)
            output = torch.cat((gate*torch.cos(p), gate*torch.sin(p), (1-gate)*g), dim=2)
        return output

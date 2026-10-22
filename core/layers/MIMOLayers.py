import torch.nn as nn
from core.layers.MIMOSublayers import FeedForward, MultiHeadAttention_S
# from core.layers.FANLayer import FANLayer
import torch





class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads):
        super().__init__()
        self.d_model = d_model   # 128
        self.norm_1 = nn.GroupNorm(1, d_model)
        self.norm_2 = nn.GroupNorm(1, d_model)
        self.attn_1 = MultiHeadAttention_S(heads, d_model)
        # self.mlp1 = FANLayer(input_dim=self.d_model, output_dim=self.d_model, with_gate=True)
        # self.mlp2 = FANLayer(input_dim=self.d_model, output_dim=self.d_model, with_gate=True)
        self.ff = FeedForward(d_model)

    def forward(self, x, LK):
        b, s, c, h, w = x.size()  # 12 5 128 16 16
        x = x + self.attn_1(x, x, x, LK)
        x = self.ff(x)
        x = self.norm_1(x.contiguous().view(-1, c, h, w)).view(b, s, c, h, w)
        x = x + self.ff(x)
        x = self.norm_2(x.view(-1, c, h, w)).view(b, s, c, h, w)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads):
        super().__init__()
        self.norm_1 = nn.GroupNorm(1, d_model)
        self.norm_2 = nn.GroupNorm(1, d_model)
        self.norm_3 = nn.GroupNorm(1, d_model)
        self.attn_1 = MultiHeadAttention_S(heads, d_model)
        self.attn_2 = MultiHeadAttention_S(heads, d_model)
        # self.mlp = FANLayer(input_dim=d_model, output_dim=d_model, with_gate=True)
        self.ff = FeedForward(d_model)

    def forward(self, x, e_outputs, LK):
        b, s, c, h, w = x.size()

        x = x + self.attn_1(x, x, x, LK)
        x = self.norm_1(x.contiguous().view(-1, c, h, w)).view(b, s, c, h, w)

        x = x + self.attn_2(x, e_outputs, e_outputs, LK)
        x = self.norm_2(x.view(-1, c, h, w)).view(b, s, c, h, w)

        x = x + self.ff(x)
        x = self.norm_3(x.view(-1, c, h, w)).view(b, s, c, h, w)

        return x

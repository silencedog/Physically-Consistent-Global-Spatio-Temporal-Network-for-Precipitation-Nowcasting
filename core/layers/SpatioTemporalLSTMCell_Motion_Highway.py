import torch
import torch.nn as nn
from core.layers.MIMOModels import Transformer

class SpatioTemporalLSTMCell(nn.Module):
    def __init__(self, configs, in_channel, num_hidden, height, width, filter_size, stride):
        super(SpatioTemporalLSTMCell, self).__init__()

        self.num_hidden = num_hidden
        self._forget_bias = 1.0
        padding = filter_size // 2
        self.ViT = Transformer(configs)

        self.conv_x = nn.Sequential(
            nn.Conv2d(in_channel, num_hidden * 7, filter_size, stride, padding, bias=False),
            nn.LayerNorm([num_hidden * 7, width, width])
        )
        self.conv_h = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden * 4, filter_size, stride, padding, bias=False),
            nn.LayerNorm([num_hidden * 4, width, width])
        )
        self.conv_m = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden * 3, filter_size, stride, padding, bias=False),
            nn.LayerNorm([num_hidden * 3, width, width])
        )
        self.conv_o = nn.Sequential(
            nn.Conv2d(num_hidden * 2, num_hidden, filter_size, stride, padding, bias=False),
            nn.LayerNorm([num_hidden, width, width])
        )

        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x_t, h_t, c_t, m_t, motion_highway, LK_expanded):
        h_t, c_t, m_t = h_t.cuda(), c_t.cuda(), m_t.cuda()   # 12 64 32 32
        x_concat = self.conv_x(x_t)  # 12 448 32 32
        h_concat = self.conv_h(h_t)  # 12 256 32 32
        m_concat = self.conv_m(m_t)  # 12 192 32 32
        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = torch.split(x_concat, self.num_hidden, dim=1) # 12 64 32 32
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)  # 12 64 32 32
        i_m, f_m, g_m = torch.split(m_concat, self.num_hidden, dim=1)  # 12 64 32 32

        i_global = self.ViT(torch.stack([i_x_prime, i_h], dim=1), LK_expanded)
        i_x_prime = i_global[:, 0, ]
        i_h = i_global[:, 1, ]

        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h + self._forget_bias)
        g_t = torch.tanh(g_x + g_h)
        c_new = f_t * c_t + i_t * g_t

        i_t_prime = torch.sigmoid(i_x_prime + i_m)
        f_t_prime = torch.sigmoid(f_x_prime + f_m + self._forget_bias)
        g_t_prime = torch.tanh(g_x_prime + g_m)
        m_new = f_t_prime * m_t + i_t_prime * g_t_prime

        mem = torch.cat((c_new, m_new), 1)
        m_new_new = self.conv_last(mem)

        o_t = torch.sigmoid(o_x + o_h + self.conv_o(mem))
        h_new = o_t * torch.tanh(m_new_new) + (1 - o_t) * motion_highway
        motion_highway = h_new
        return h_new, c_new, m_new, motion_highway
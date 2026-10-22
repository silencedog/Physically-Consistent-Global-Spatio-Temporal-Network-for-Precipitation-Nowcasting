import torch
import torch.nn as nn
import torch.nn.functional as F
from core.layers.DifferentiableLK import differentiable_LK_function
from core.layers.SpatioTemporalLSTMCell_Motion_Highway import SpatioTemporalLSTMCell


class Network(nn.Module):
    def __init__(self, configs):
        super(Network, self).__init__()
        self.configs = configs
        self.patch_height = configs.img_height // configs.patch_size
        self.patch_width = configs.img_width // configs.patch_size
        self.patch_ch = configs.img_channel * (configs.patch_size ** 2)
        self.num_layers = configs.num_layers
        self.num_hidden = configs.num_hidden
        self.neighbour = 3
        self.motion_hidden = 2 * self.neighbour * self.neighbour

        cell_list = []
        for i in range(self.num_layers):
            in_channel = self.patch_ch if i == 0 else self.num_hidden
            cell_list.append(
                SpatioTemporalLSTMCell(self.configs, in_channel, self.num_hidden, self.patch_height, self.patch_width,
                                       configs.filter_size, configs.stride)
            )
        enc_list = []
        for i in range(self.num_layers - 1):
            enc_list.append(
                nn.Conv2d(self.num_hidden, self.num_hidden // 4, kernel_size=configs.filter_size, stride=2,
                          padding=configs.filter_size // 2),
            )
        dec_list = []
        for i in range(self.num_layers - 1):
            dec_list.append(
                nn.ConvTranspose2d(self.num_hidden // 4, self.num_hidden, kernel_size=4, stride=2,
                                   padding=1),
            )
        gate_list = []
        for i in range(self.num_layers - 1):
            gate_list.append(
                nn.Conv2d(self.num_hidden * 2, self.num_hidden, kernel_size=configs.filter_size, stride=1,
                          padding=configs.filter_size // 2),
            )

        self.gate_list = nn.ModuleList(gate_list)
        self.cell_list = nn.ModuleList(cell_list)
        self.enc_list = nn.ModuleList(enc_list)
        self.dec_list = nn.ModuleList(dec_list)
        self.conv_last = nn.Conv2d(self.num_hidden, self.patch_ch, 1, stride=1, padding=0, bias=False)
        self.conv_first_v = nn.Conv2d(self.patch_ch, self.num_hidden, 1, stride=1, padding=0, bias=False)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=4, stride=4, return_indices=False)

        # 根据你的描述，原始的LK_expanded是 [6, 2, 16, 16, 16]
        # 我们创建一个专门的LK投影层，生成正确的形状
        # 2可能是头部数或时间维度，16是隐藏维
        self.lk_projection_1 = nn.Sequential(
            nn.Conv2d(4, 64, 1, 1, 0, bias=False),
            nn.GroupNorm(1, 64),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d((16, 16)),  # 将空间尺寸调整为16x16
        )

        # 然后将64通道转换为32通道（2×16）
        self.lk_projection_2 = nn.Sequential(
            nn.Conv2d(64, 32, 1, 1, 0, bias=False),
            nn.GroupNorm(1, 32),
            nn.SiLU(inplace=True),
        )

    def forward(self, all_frames, mask_true, batch_size):
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        frames = all_frames
        mask_true = mask_true
        dis_frames = []
        next_frames = []
        final_frames = []
        h_t = []
        c_t = []
        h_t_conv = []
        h_t_conv_offset = []
        mean = []

        for i in range(self.num_layers):
            zeros = torch.zeros(
                [batch_size, self.num_hidden, self.patch_height, self.patch_width],
                device=frames.device)
            h_t.append(zeros)
            c_t.append(zeros)

        for i in range(self.num_layers - 1):
            zeros = torch.zeros(
                [batch_size, self.num_hidden // 4, self.patch_height // 2,
                 self.patch_width // 2],
                device=frames.device)
            h_t_conv.append(zeros)
            zeros = torch.zeros(
                [batch_size, self.motion_hidden, self.patch_height // 2, self.patch_width // 2],
                device=frames.device)
            h_t_conv_offset.append(zeros)
            mean.append(zeros)

        mem = torch.zeros([batch_size, self.num_hidden, self.patch_height, self.patch_width],
                          device=frames.device)
        motion_highway = torch.zeros(
            [batch_size, self.num_hidden, self.patch_height, self.patch_width],
            device=frames.device)

        # 初始化前一帧
        prev_frame = None

        for t in range(self.configs.total_length - 1):
            if t < self.configs.input_length:
                net = frames[:, t]
            else:
                net = mask_true[:, t - self.configs.input_length] * frames[:, t] + \
                      (1 - mask_true[:, t - self.configs.input_length]) * x_gen

            # 可微分LK光流计算部分
            if t == 0:
                LK_expanded = 0
            else:
                # 获取当前帧
                current_frame = net
                # 使用可微分LK光流计算
                # 创建一个包含两帧的序列 [B, 2, H, W, C]
                # 先将格式从 [B, H, W, C] 转换为 [B, C, H, W]
                prev_frame_tensor = prev_frame.permute(0, 3, 1, 2)  # [B, C, H, W]
                current_frame_tensor = current_frame.permute(0, 3, 1, 2)  # [B, C, H, W]
                frame_pair = torch.stack([prev_frame_tensor, current_frame_tensor], dim=1)
                # 计算光流（可微分）
                moved_frames, flow_fields = differentiable_LK_function(frame_pair)
                # 获取光流 [B, 2, H, W]
                flow = flow_fields[:, 0]
                # 扩展为4通道 [B, 4, H, W]
                LK_4channel = torch.cat([flow, flow], dim=1)

                # 第一步：将4通道转换为64通道，调整空间尺寸
                LK_64channel = self.lk_projection_1(LK_4channel)  # [B, 64, 16, 16]

                # 第二步：将64通道转换为32通道
                LK_32channel = self.lk_projection_2(LK_64channel)  # [B, 32, 16, 16]

                # 现在我们需要将 [B, 32, 16, 16] 转换为 [B, 2, 16, 16, 16]
                # 32 = 2 × 16，所以可以直接reshape
                LK_reshaped = LK_32channel.view(batch_size, 2, 16, 16, 16)

                # 将形状调整为 [B, 2, 16, 16, 16]
                LK_expanded = LK_reshaped.contiguous()

            # 更新前一帧为当前帧，用于下一时间步
            prev_frame = net.clone()

            motion_highway = self.conv_first_v(net)

            # 将LK_expanded传递给cell
            # 注意：我们需要确保LK_expanded的形状是 [B, 2, 16, H, W]
            h_t[0], c_t[0], mem, motion_highway = self.cell_list[0](net, h_t[0], c_t[0], mem, motion_highway,
                                                                    LK_expanded)

            if self.num_layers > 1:
                net = self.enc_list[0](h_t[0])
                h_t_tmp = self.dec_list[0](net)
                o_t = torch.sigmoid(self.gate_list[0](torch.cat([h_t_tmp, h_t[0]], dim=1)))
                h_t[0] = o_t * h_t_tmp + (1 - o_t) * h_t[0]

                for i in range(1, self.num_layers - 1):
                    h_t[i], c_t[i], mem, motion_highway = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], mem,
                                                                            motion_highway,
                                                                            LK_expanded)
                    net = self.enc_list[i](h_t[i])
                    h_t_tmp = self.dec_list[i](net)
                    o_t = torch.sigmoid(self.gate_list[i](torch.cat([h_t_tmp, h_t[i]], dim=1)))
                    h_t[i] = o_t * h_t_tmp + (1 - o_t) * h_t[i]

            # 最后一层
            last_layer_idx = self.num_layers - 1
            if self.num_layers > 1:
                h_t[last_layer_idx], c_t[last_layer_idx], mem, motion_highway = self.cell_list[last_layer_idx](
                    h_t[last_layer_idx - 1], h_t[last_layer_idx], c_t[last_layer_idx], mem, motion_highway,
                    LK_expanded)
            else:
                # 如果只有一层，使用输入net
                h_t[last_layer_idx], c_t[last_layer_idx], mem, motion_highway = self.cell_list[last_layer_idx](
                    net, h_t[last_layer_idx], c_t[last_layer_idx], mem, motion_highway,
                    LK_expanded)

            dis_frames.append(h_t[last_layer_idx])
            x_gen = self.conv_last(h_t[last_layer_idx])
            next_frames.append(x_gen)

        # [length, batch, channel, height, width] -> [batch, length, channel, height, width]
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 2, 3, 4).contiguous()
        final_frames.append(next_frames)

        return final_frames
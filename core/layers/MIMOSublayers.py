import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PhysicsConstrainedLK(nn.Module):  # 继承Module确保设备兼容
    def __init__(self, max_divergence=0.1, alpha=0.3):
        super().__init__()
        self.max_divergence = max_divergence
        self.alpha = alpha
        self.register_buffer('prev_flow', None)  # 正确注册缓冲区

    def _apply_divergence_constraint(self, flow_field):
        """可微分物理约束实现"""
        # 移除所有clone/detach操作，保持计算图连通
        H, W = flow_field.shape[:2]

        dx = flow_field[1:, :, 0] - flow_field[:-1, :, 0]
        dy = flow_field[:, 1:, 1] - flow_field[:, :-1, 1]

        # 保持原有维度处理逻辑nbshu
        trunc_h = min(dx.size(0), dy.size(0))
        trunc_w = min(dx.size(1), dy.size(1))

        # 构建可微掩码系统
        divergence = (dx[:trunc_h, :trunc_w] + dy[:trunc_h, :trunc_w]) / 2.0
        mask = (divergence.abs() > self.max_divergence).float()  # 转为浮点掩码

        # 可微分修正操作
        correction = (1 - 0.1 * mask)
        flow_field = flow_field.clone()  # 此处clone不会阻断计算图
        flow_field[:trunc_h, :trunc_w] *= correction.unsqueeze(-1)

        return flow_field

    def _temporal_smoothing(self, current_flow):
        """可微分时间平滑"""
        if self.prev_flow is None:
            self.prev_flow = current_flow.detach()  # 仅初始化时阻断梯度
            return current_flow

            # 自动广播安全的插值
        if self.prev_flow.shape != current_flow.shape:
            self.prev_flow = F.interpolate(
                self.prev_flow.permute(2, 0, 1).unsqueeze(0),
                size=current_flow.shape[:2],
                mode='bilinear',
                align_corners=False
            ).squeeze(0).permute(1, 2, 0)

        smoothed = torch.lerp(self.prev_flow, current_flow, self.alpha)
        self.prev_flow = smoothed.detach()  # 防止历史梯度累积
        return smoothed

def attention_s(q, k, v):
    # q.view(q.size(0), q.size(1), -1) = 192 5 4096
    """scores_c = torch.matmul(q.view(q.size(0), q.size(1), -1), k.view(k.size(0), k.size(1), -1).permute(0, 2, 1)) \
             / math.sqrt(q.size(2) * q.size(3) * q.size(4))   # 192 5 5
    scores_c = torch.sigmoid(scores_c)
    v_c = torch.matmul(scores_c, v.reshape(v.size(0), v.size(1), -1))  # 192 5 4096
    output_c = v_c.reshape(q.size()) # 192 5 16 16 16"""
    # B FC HW -> B HW HW
    # 创建两个独立处理器实例
    physics_channel = PhysicsConstrainedLK()  # 处理通道特征
    physics_spatial = PhysicsConstrainedLK() # 处理空间特征

    q_hw = q.contiguous().view(q.size(0), -1, q.size(3), q.size(4))  # (Batch, Frames*Channels, Width, Height)
    k_hw = k.contiguous().view(k.size(0), -1, k.size(3), k.size(4))  # (Batch, Frames*Channels, Width, Height)
    v_hw = v.contiguous().view(v.size(0), -1, v.size(3), v.size(4))  # (Batch, Frames*Channels, Width, Height)

    scores_c = torch.matmul(q_hw.view(q_hw.size(0), q_hw.size(1), -1).permute(0, 2, 1), k_hw.view(k_hw.size(0), k_hw.size(1), -1)) \
               / math.sqrt(q_hw.size(2) * q_hw.size(3))

    constrained_flow_c = physics_channel._apply_divergence_constraint(scores_c)
    scores_c = physics_channel._temporal_smoothing(constrained_flow_c)

    scores_c = torch.sigmoid(scores_c)
    v_c = torch.matmul(v_hw.view(v_hw.size(0), v_hw.size(1), -1), scores_c)
    output_c = v_c.view(q.size())

    # 将 q, k 和 v 视图展平为 (Batch, Frames, Channels, Width * Height)
    q_flat = q.contiguous().view(q.size(0), q.size(1), q.size(2), -1)  # (Batch, Frames, Channels, Width*Height)
    k_flat = k.contiguous().view(k.size(0), k.size(1), k.size(2), -1)  # (Batch, Frames, Channels, Width*Height)
    v_flat = v.contiguous().view(v.size(0), v.size(1), v.size(2), -1)  # (Batch, Frames, Channels, Width*Height)


    scores_s = torch.matmul(q_flat.view(q_flat.size(0), -1, q_flat.size(3)),
                            k_flat.view(k_flat.size(0), -1, k_flat.size(3)).permute(0, 2, 1)) / math.sqrt(q.size(2))


    constrained_flow_s = physics_spatial._apply_divergence_constraint(scores_s)
    scores_s = physics_spatial._temporal_smoothing(constrained_flow_s)

        # 使用 sigmoid 激活函数将分数映射到 [0, 1] 范围
    scores_s = torch.sigmoid(scores_s)  # 计算的得分大小，范围 [0, 1]

    # 使用注意力得分对 v 进行加权
    v_s = torch.matmul(scores_s,
                       v_flat.view(v_flat.size(0), -1, v_flat.size(3)))  # (Batch, Frames, Width*Height, Channels)

    # 将加权后的结果恢复回原始的空间形状
    output_s = v_s.view(q.size())  # (Batch, Frames, Channels, Width, Height)

    return output_s + output_c

class MultiHeadAttention_S(nn.Module):
    def __init__(self, heads, d_model):
        super().__init__()
        self.d_model = d_model
        self.h = heads

        self.q_Conv = nn.Sequential(nn.Conv2d(self.d_model, self.d_model, 1, 1, 0, bias=False),
                                    nn.GroupNorm(1, self.d_model),
                                    )
        self.v_Conv = nn.Sequential(nn.Conv2d(self.d_model, self.d_model, 1, 1, 0, bias=False),
                                    nn.GroupNorm(1, self.d_model),
                                    )
        self.k_Conv = nn.Sequential(nn.Conv2d(self.d_model, self.d_model, 1, 1, 0, bias=False),
                                    nn.GroupNorm(1, self.d_model),
                                    )
        self.v_post_f = nn.Sequential(
            nn.Conv2d(d_model, d_model, 1, 1, 0, bias=False),
            nn.GroupNorm(1, d_model),
            nn.SiLU(inplace=True),
        )

    def forward(self, q, k, v, LK):
        q = q * LK + q
        k = k * LK + k
        v = v * LK + v
        b_q, s_q, c_q, h_q, w_q = q.size()
        b_k, s_k, c_k, h_k, w_k = k.size()
        b_v, s_v, c_v, h_v, w_v = v.size()
        q = self.q_Conv(q.view(q.size(0) * q.size(1), *q.shape[2:])).view(q.size(0)*q.size(1), self.h,
                                                                                self.d_model // self.h, h_q, w_q)
        q = q.view(b_q, s_q, self.h, self.d_model // self.h, h_q, w_q).permute(0, 2, 1, 3, 4, 5)
        q = q.contiguous().view(q.size(0)*q.size(1), *q.shape[2:])

        k = self.k_Conv(k.view(k.size(0) * k.size(1), *k.shape[2:])).view(k.size(0) * k.size(1), self.h,
                                                                                self.d_model // self.h, h_q, w_q)
        k = k.view(b_k, s_k, self.h, self.d_model // self.h, h_q, w_q).permute(0, 2, 1, 3, 4, 5)
        k = k.contiguous().view(k.size(0) * k.size(1), *k.shape[2:])

        v = self.v_Conv(v.view(v.size(0) * v.size(1), *v.shape[2:])).view(v.size(0) * v.size(1), self.h,
                                                                                self.d_model // self.h, h_q, w_q)
        v = v.view(b_v, s_v, self.h, self.d_model // self.h, h_q, w_q).permute(0, 2, 1, 3, 4, 5)
        v = v.contiguous().view(v.size(0) * v.size(1), *v.shape[2:])

        output = attention_s(q, k, v).view(b_q, self.h, s_q, self.d_model // self.h, h_q, w_q).permute(0, 2, 1, 3, 4, 5)
        output = self.v_post_f(output.contiguous().view(b_q*s_q, self.h, self.d_model // self.h,
                                              h_q, w_q).view(b_q*s_q, self.d_model, h_q, w_q)).view(b_q, s_q, c_q, h_q, w_q)

        return output

class MultiHeadAttention_D(nn.Module):
    def __init__(self, heads, d_model):
        super().__init__()
        self.d_model = d_model
        self.h = heads

        self.q_Conv = nn.Sequential(nn.Conv2d(self.d_model, self.d_model, 1, 1, 0, bias=False),
                                    nn.GroupNorm(1, self.d_model),
                                    )
        self.v_Conv = nn.Sequential(nn.Conv2d(self.d_model, self.d_model, 1, 1, 0, bias=False),
                                    nn.GroupNorm(1, self.d_model),
                                    )
        self.k_Conv = nn.Sequential(nn.Conv2d(self.d_model, self.d_model, 1, 1, 0, bias=False),
                                    nn.GroupNorm(1, self.d_model),
                                    )
        self.v_post_f = nn.Sequential(
            nn.Conv2d(d_model, d_model, 1, 1, 0, bias=False),
            nn.GroupNorm(1, d_model),
            nn.SiLU(inplace=True),
        )

    def forward(self, q, k, v):
        b_q, s_q, c_q, h_q, w_q = q.size()
        b_k, s_k, c_k, h_k, w_k = k.size()
        b_v, s_v, c_v, h_v, w_v = v.size()
        q = self.q_Conv(q.view(q.size(0) * q.size(1), *q.shape[2:])).view(q.size(0)*q.size(1), self.h,
                                                                                self.d_model // self.h, h_q, w_q)
        q = q.view(b_q, s_q, self.h, self.d_model // self.h, h_q, w_q).permute(0, 2, 1, 3, 4, 5)
        q = q.view(q.size(0)*q.size(1), *q.shape[2:])

        k = self.k_Conv(k.view(k.size(0) * k.size(1), *k.shape[2:])).view(k.size(0) * k.size(1), self.h,
                                                                                self.d_model // self.h, h_q, w_q)
        k = k.view(b_k, s_k, self.h, self.d_model // self.h, h_q, w_q).permute(0, 2, 1, 3, 4, 5)
        k = k.view(k.size(0) * k.size(1), *k.shape[2:])

        v = self.v_Conv(v.view(v.size(0) * v.size(1), *v.shape[2:])).view(v.size(0) * v.size(1), self.h,
                                                                                self.d_model // self.h, h_q, w_q)
        v = v.view(b_v, s_v, self.h, self.d_model // self.h, h_q, w_q).permute(0, 2, 1, 3, 4, 5)
        v = v.view(v.size(0) * v.size(1), *v.shape[2:])

        output = attention_s(q, k, v).view(b_q, self.h, s_q, self.d_model // self.h, h_q, w_q).permute(0, 2, 1, 3, 4, 5)
        output = self.v_post_f(output.view(b_q*s_q, self.h, self.d_model // self.h,
                                              h_q, w_q).view(b_q*s_q, self.d_model, h_q, w_q)).view(b_q, s_q, c_q, h_q, w_q)

        return output

class FeedForward11(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.ff1 = nn.Sequential(
            nn.Conv3d(d_model, 2 * d_model, [3,3,3], [2,1,1], 1, bias=False),
            nn.GroupNorm(1, 2 * d_model),
            nn.SiLU(inplace=True),
        )
        self.ff2 = nn.Sequential(
            nn.Conv3d(2 * d_model, d_model, 3, 1, 1, bias=False),
            nn.GroupNorm(1, d_model),
            nn.SiLU(inplace=True),
        )

    def forward(self, x, e_outputs, e_outputs1):
        XX = torch.cat([x, e_outputs], dim=1)
        x = self.ff2(self.ff1(XX.permute(0, 2, 1, 3, 4))).permute(0, 2, 1, 3, 4)
        return x

class FeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.ff1 = nn.Sequential(
            nn.Conv3d(d_model, 2 * d_model, 3, 1, 1, bias=False),
            nn.GroupNorm(1, 2 * d_model),
            nn.SiLU(inplace=True),
        )
        self.ff2 = nn.Sequential(
            nn.Conv3d(2 * d_model, d_model, 3, 1, 1, bias=False),
            nn.GroupNorm(1, d_model),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        x = self.ff2(self.ff1(x.permute(0, 2, 1, 3, 4))).permute(0, 2, 1, 3, 4)
        return x
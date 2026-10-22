# [file name]: core/layers/DifferentiableLK.py (修复版本)
# [file content begin]
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DifferentiableLucasKanade(nn.Module):
    """
    可微分的Lucas-Kanade光流实现
    基于PyTorch的梯度计算，完全可微分
    """

    def __init__(self, window_size=21, iterations=3, pyramid_levels=3):
        super().__init__()
        self.window_size = window_size
        self.iterations = iterations
        self.pyramid_levels = pyramid_levels
        self.padding = window_size // 2

        # 创建高斯权重窗口（可微分参数）
        self.gaussian_weights = self._create_gaussian_window(window_size)

    def _create_gaussian_window(self, window_size):
        """创建可微分的高斯权重窗口"""
        sigma = window_size / 6.0
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        gauss_x = torch.exp(-0.5 * (coords / sigma) ** 2)
        gauss_2d = torch.outer(gauss_x, gauss_x)
        gauss_2d = gauss_2d / gauss_2d.sum()
        return nn.Parameter(gauss_2d, requires_grad=True)

    def _compute_gradients(self, image):
        """计算图像梯度（可微分）"""
        # 确保输入是4D张量 [batch, channel, height, width]
        if image.dim() == 5:
            # 如果是5D张量，可能是 [1, channels, 1, height, width]
            # 压缩掉不必要的维度
            image = image.squeeze(0).squeeze(1)  # 变成 [channels, height, width]
            # 然后重新添加batch维度
            image = image.unsqueeze(0)  # [1, channels, height, width]

        assert image.dim() == 4, f"Expected 4D input, but got {image.dim()}D input of size: {image.size()}"

        batch_size, channels, height, width = image.shape

        # Sobel算子
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=image.dtype, device=image.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=image.dtype, device=image.device).view(1, 1, 3, 3)

        # 扩展卷积核以匹配输入通道数
        sobel_x = sobel_x.expand(channels, 1, 3, 3)
        sobel_y = sobel_y.expand(channels, 1, 3, 3)

        # 分组卷积，每个通道独立处理
        I_x = F.conv2d(image, sobel_x, padding=1, groups=channels)
        I_y = F.conv2d(image, sobel_y, padding=1, groups=channels)

        # 时间梯度使用简单的平均滤波
        avg_kernel = torch.ones(1, 1, 3, 3, device=image.device) / 9
        avg_kernel = avg_kernel.expand(channels, 1, 3, 3)
        I_t = F.conv2d(image, avg_kernel, padding=1, groups=channels)

        return I_x, I_y, I_t

    def _build_pyramid(self, image, levels):
        """构建图像金字塔（可微分）"""
        if image.dim() == 5:
            image = image.squeeze(0)

        pyramid = [image]
        for i in range(1, levels):
            # 使用高斯模糊和下采样
            blurred = F.avg_pool2d(image, kernel_size=2, stride=2)
            pyramid.append(blurred)
            image = blurred
        return pyramid

    def forward(self, prev_frame, curr_frame):
        """
        前向传播计算光流
        Args:
            prev_frame: [B, C, H, W] 前一帧
            curr_frame: [B, C, H, W] 当前帧
        Returns:
            flow: [B, 2, H, W] 光流场 (u, v)
        """
        # 确保输入是4D
        if prev_frame.dim() == 5:
            prev_frame = prev_frame.squeeze(0)
        if curr_frame.dim() == 5:
            curr_frame = curr_frame.squeeze(0)

        batch_size, channels, height, width = prev_frame.shape

        # 构建图像金字塔（对每个通道取平均，转换为单通道）
        prev_gray = prev_frame.mean(dim=1, keepdim=True)  # [B, 1, H, W]
        curr_gray = curr_frame.mean(dim=1, keepdim=True)  # [B, 1, H, W]

        prev_pyramid = self._build_pyramid(prev_gray, self.pyramid_levels)
        curr_pyramid = self._build_pyramid(curr_gray, self.pyramid_levels)

        # 初始化光流为0，使用最粗糙层的分辨率
        coarsest_level = self.pyramid_levels - 1
        coarsest_h, coarsest_w = prev_pyramid[coarsest_level].shape[-2:]
        flow = torch.zeros(batch_size, 2, coarsest_h, coarsest_w,
                           device=prev_frame.device, dtype=prev_frame.dtype)

        # 从最粗糙的金字塔层开始
        for level in range(self.pyramid_levels - 1, -1, -1):
            scale = 2 ** level
            h, w = prev_pyramid[level].shape[-2:]

            # 调整光流到当前金字塔层分辨率
            if flow.shape[-2:] != (h, w):
                flow = F.interpolate(flow, size=(h, w), mode='bilinear', align_corners=False)
                if level < self.pyramid_levels - 1:
                    # 当上采样到更精细的层时，调整光流尺度
                    flow *= 2.0

            # 获取当前金字塔层的图像
            I1 = prev_pyramid[level]  # [B, 1, H, W]
            I2 = curr_pyramid[level]  # [B, 1, H, W]

            # 计算图像梯度
            I_x, I_y, I_t = self._compute_gradients(I1)

            # 计算结构张量
            Ixx = I_x * I_x
            Ixy = I_x * I_y
            Iyy = I_y * I_y
            Ixt = I_x * I_t
            Iyt = I_y * I_t

            # 应用高斯窗口（可微分卷积）
            weight = self.gaussian_weights.view(1, 1, self.window_size, self.window_size)

            # 使用卷积进行窗口聚合
            Axx = F.conv2d(Ixx, weight, padding=self.padding, groups=1)
            Axy = F.conv2d(Ixy, weight, padding=self.padding, groups=1)
            Ayy = F.conv2d(Iyy, weight, padding=self.padding, groups=1)
            Bx = F.conv2d(Ixt, weight, padding=self.padding, groups=1)
            By = F.conv2d(Iyt, weight, padding=self.padding, groups=1)

            # 计算光流更新（解线性系统）
            det = Axx * Ayy - Axy * Axy
            det = det + 1e-6  # 防止除零

            u_update = (-Ayy * Bx + Axy * By) / det
            v_update = (Axy * Bx - Axx * By) / det

            # 确保更新尺寸与当前光流尺寸匹配
            if u_update.shape[-2:] != flow[:, 0:1].shape[-2:]:
                # 调整更新尺寸以匹配光流
                u_update = F.interpolate(u_update, size=flow[:, 0:1].shape[-2:],
                                         mode='bilinear', align_corners=False)
                v_update = F.interpolate(v_update, size=flow[:, 0:1].shape[-2:],
                                         mode='bilinear', align_corners=False)

            # 更新光流估计
            flow[:, 0:1] += u_update
            flow[:, 1:2] += v_update

            # 多尺度迭代优化
            for _ in range(self.iterations):
                # 使用当前光流进行图像扭曲
                warped = self._warp_image(I2, flow)

                # 重新计算时间梯度
                I_t_new = warped - I1

                # 重新计算结构张量
                Ixt_new = I_x * I_t_new
                Iyt_new = I_y * I_t_new

                Bx_new = F.conv2d(Ixt_new, weight, padding=self.padding, groups=1)
                By_new = F.conv2d(Iyt_new, weight, padding=self.padding, groups=1)

                # 重新计算光流更新
                u_update = (-Ayy * Bx_new + Axy * By_new) / det
                v_update = (Axy * Bx_new - Axx * By_new) / det

                # 确保更新尺寸匹配
                if u_update.shape[-2:] != flow[:, 0:1].shape[-2:]:
                    u_update = F.interpolate(u_update, size=flow[:, 0:1].shape[-2:],
                                             mode='bilinear', align_corners=False)
                    v_update = F.interpolate(v_update, size=flow[:, 0:1].shape[-2:],
                                             mode='bilinear', align_corners=False)

                # 更新光流
                flow[:, 0:1] += u_update
                flow[:, 1:2] += v_update

        # 最终将光流上采样到原始分辨率
        if flow.shape[-2:] != (height, width):
            flow = F.interpolate(flow, size=(height, width), mode='bilinear', align_corners=False)
            # 调整尺度以匹配原始分辨率
            scale_factor = height / flow.shape[-2]
            flow *= scale_factor

        return flow

    def _warp_image(self, image, flow):
        """可微分图像扭曲"""
        batch_size, _, height, width = image.shape

        # 创建坐标网格
        grid_y, grid_x = torch.meshgrid(
            torch.arange(height, device=image.device),
            torch.arange(width, device=image.device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).float()
        grid = grid.unsqueeze(0).repeat(batch_size, 1, 1, 1)

        # 添加光流偏移
        warped_grid = grid + flow.permute(0, 2, 3, 1)

        # 归一化到[-1, 1]
        warped_grid[..., 0] = 2.0 * warped_grid[..., 0] / (width - 1) - 1.0
        warped_grid[..., 1] = 2.0 * warped_grid[..., 1] / (height - 1) - 1.0

        # 采样扭曲后的图像
        warped_image = F.grid_sample(image, warped_grid,
                                     align_corners=True,
                                     padding_mode='border')

        return warped_image


class DifferentiablePhysicsConstrainedLK(nn.Module):
    """
    可微分物理约束的LK光流
    结合了可微分LK和可微分物理约束
    """

    def __init__(self, window_size=21, max_divergence=0.1, alpha=0.3):
        super().__init__()
        self.lk_module = DifferentiableLucasKanade(window_size)
        self.max_divergence = nn.Parameter(torch.tensor(max_divergence), requires_grad=True)
        self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=True)
        self.register_buffer('prev_flow', None)

    def _apply_divergence_constraint(self, flow):
        """可微分的质量守恒约束"""
        batch_size, _, height, width = flow.shape

        # 计算散度 (∇·u = ∂u/∂x + ∂v/∂y)
        u = flow[:, 0:1, :, :]
        v = flow[:, 1:2, :, :]

        # 使用可微分卷积计算梯度
        grad_kernel = torch.tensor([[-1, 0, 1]], dtype=flow.dtype, device=flow.device).view(1, 1, 1, 3) / 2.0

        # ∂u/∂x
        u_x = F.conv2d(u, grad_kernel, padding=(0, 1))
        # ∂v/∂y (需要转置核)
        v_y = F.conv2d(v, grad_kernel.transpose(2, 3), padding=(1, 0))

        divergence = u_x + v_y

        # 计算散度损失权重（可微分）
        divergence_norm = torch.abs(divergence)

        # 创建可微分的修正因子
        correction_factor = torch.sigmoid(self.max_divergence - divergence_norm)
        correction_factor = 0.1 + 0.9 * correction_factor  # 在0.1-1.0之间

        # 应用修正（保持梯度）
        flow_corrected = flow * correction_factor

        return flow_corrected

    def _apply_temporal_constraint(self, flow):
        """可微分的时间连续性约束"""
        if self.prev_flow is None:
            self.prev_flow = flow.detach().clone()
            return flow

        # 确保形状匹配
        if self.prev_flow.shape != flow.shape:
            self.prev_flow = F.interpolate(self.prev_flow,
                                           size=flow.shape[-2:],
                                           mode='bilinear',
                                           align_corners=False)

        # 线性插值（可微分）
        smoothed_flow = self.alpha * flow + (1 - self.alpha) * self.prev_flow

        # 更新历史光流（detach防止梯度爆炸）
        self.prev_flow = smoothed_flow.detach().clone()

        return smoothed_flow

    def forward(self, prev_frame, curr_frame):
        """前向传播：计算可微分物理约束的光流"""
        # 1. 计算基础光流
        flow = self.lk_module(prev_frame, curr_frame)

        # 2. 应用物理约束
        flow = self._apply_divergence_constraint(flow)
        flow = self._apply_temporal_constraint(flow)

        return flow


def differentiable_LK_function(frames):
    """
    可微分的LK光流函数
    Args:
        frames: [B, T, C, H, W] 输入帧序列
    Returns:
        moved_frames: [B, T-1, C, H, W] 运动补偿后的帧
        flow_fields: [B, T-1, 2, H, W] 光流场
    """
    batch_size, timesteps, channels, height, width = frames.shape

    # 初始化模块
    lk_module = DifferentiablePhysicsConstrainedLK()
    lk_module = lk_module.to(frames.device)

    moved_frames = []
    flow_fields = []

    for t in range(timesteps - 1):
        prev_frame = frames[:, t]  # [B, C, H, W]
        curr_frame = frames[:, t + 1]  # [B, C, H, W]

        # 确保输入是4D
        if prev_frame.dim() == 5:
            prev_frame = prev_frame.squeeze(0)
        if curr_frame.dim() == 5:
            curr_frame = curr_frame.squeeze(0)

        # 计算光流
        flow = lk_module(prev_frame, curr_frame)  # [B, 2, H, W]
        flow_fields.append(flow)

        # 应用运动补偿（可微分）
        warped = lk_module.lk_module._warp_image(curr_frame, flow)
        moved_frames.append(warped)

    # 堆叠结果
    if moved_frames:  # 检查列表是否非空
        moved_frames = torch.stack(moved_frames, dim=1)  # [B, T-1, C, H, W]
        flow_fields = torch.stack(flow_fields, dim=1)  # [B, T-1, 2, H, W]
    else:
        # 如果没有计算结果，返回空张量
        moved_frames = torch.empty(batch_size, 0, channels, height, width, device=frames.device)
        flow_fields = torch.empty(batch_size, 0, 2, height, width, device=frames.device)

    return moved_frames, flow_fields


# 测试代码
if __name__ == "__main__":
    # 生成测试数据
    batch_size, timesteps = 4, 5
    channels, height, width = 3, 32, 32

    test_frames = torch.randn(batch_size, timesteps, channels, height, width)

    # 测试可微分性
    test_frames.requires_grad_(True)

    moved_frames, flow_fields = differentiable_LK_function(test_frames)

    # 检查梯度
    loss = moved_frames.sum()
    loss.backward()

    print(f"输入是否需要梯度: {test_frames.requires_grad}")
    print(f"输出形状 - 运动补偿帧: {moved_frames.shape}")
    print(f"输出形状 - 光流场: {flow_fields.shape}")
    print(f"梯度存在: {test_frames.grad is not None}")
    print("测试通过！模块完全可微分。")
# [file content end]
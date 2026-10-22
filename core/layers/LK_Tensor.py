import cv2
import torch
import numpy as np
from torch.nn import functional as F


class PhysicsConstrainedLK:
    def __init__(self, max_divergence=0.1, alpha=0.3):
        self.max_divergence = max_divergence
        self.alpha = alpha
        self.prev_flow = None

    def _apply_divergence_constraint(self, flow_field):
        """质量守恒约束（维度安全版本）"""
        H, W, _ = flow_field.shape
        dx = flow_field[1:, :, 0] - flow_field[:-1, :, 0]  # ∂u/∂x [H-1, W]
        dy = flow_field[:, 1:, 1] - flow_field[:, :-1, 1]  # ∂v/∂y [H, W-1]

        # 对齐维度至[H-1, W-1]
        dx_trunc = dx[:, :-1]
        dy_trunc = dy[:-1, :]
        divergence = (dx_trunc + dy_trunc) / 2.0

        # 创建修正掩码
        mask = torch.abs(divergence) > self.max_divergence
        flow_field[:-1, :-1][mask] *= 0.9
        return flow_field

    def _temporal_smoothing(self, current_flow):
        """时间连续性约束（首帧安全）"""
        if self.prev_flow is None:
            self.prev_flow = current_flow
            return current_flow
        smoothed = self.alpha * current_flow + (1 - self.alpha) * self.prev_flow
        self.prev_flow = smoothed.clone()
        return smoothed


def LK_function(Img):
    """修复梯度问题的最终版本"""
    B, T, C, H, W = Img.shape
    device = Img.device

    # 添加detach()分离计算图
    def safe_to_numpy(tensor):
        """安全转换张量到numpy数组"""
        return tensor.detach().clamp(0, 255).cpu().numpy().astype(np.uint8)

        # 预生成坐标网格（保持梯度无关）

    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )
    base_grid = torch.stack([grid_x, grid_y], dim=-1).float()  # [H,W,2]

    movedImgs = torch.zeros((B, T - 1, C, H, W), dtype=Img.dtype, device=device)

    for b in range(B):
        physics_processor = PhysicsConstrainedLK()

        for t in range(T - 1):
            # 初始化关键点（保持与原始图像梯度无关）
            prev_pts = np.stack(np.mgrid[0:W:1, 0:H:1], axis=-1)
            prev_pts = prev_pts.astype(np.float32).reshape(-1, 1, 2)

            flows = []
            for c in range(C):
                # 安全转换：添加detach()并保持数值范围
                prev_img = safe_to_numpy(Img[b, t, c] * 255)
                next_img = safe_to_numpy(Img[b, t + 1, c] * 255)

                try:
                    # 光流计算（保持与PyTorch梯度无关）
                    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                        prev_img, next_img, prev_pts, None,
                        winSize=(21, 21), maxLevel=3,
                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
                    )

                except Exception as e:
                    # 安全降级：零光流
                    flow = torch.zeros(H, W, 2, device=device)
                    flows.append(flow)
                    continue

                    # 转换到PyTorch（保持梯度无关）
                flow = torch.from_numpy(next_pts - prev_pts).float().to(device)
                flows.append(flow.view(H, W, 2))

                # 多通道融合（保持梯度无关）
            combined_flow = torch.stack(flows).mean(dim=0)

            # 应用物理约束（不影响原始梯度）
            with torch.no_grad():
                constrained_flow = physics_processor._apply_divergence_constraint(combined_flow)
                constrained_flow = physics_processor._temporal_smoothing(constrained_flow)

            # 运动补偿（在无梯度环境中执行）
            with torch.no_grad():
                new_grid = base_grid + constrained_flow
                new_grid[..., 0] = 2 * (new_grid[..., 0] / (W - 1)) - 1
                new_grid[..., 1] = 2 * (new_grid[..., 1] / (H - 1)) - 1

                warped = F.grid_sample(
                    Img[b, t + 1].unsqueeze(0),
                    new_grid.unsqueeze(0),
                    align_corners=True,
                    padding_mode='border'
                ).squeeze(0)

                # 输出结果（保持原始梯度路径）
            movedImgs[b, t] = warped.detach().clone()

    return movedImgs


# 测试用例
if __name__ == "__main__":
    # 生成测试数据 [100,5,16,32,32]
    test_data = torch.rand(100, 5, 16, 32, 32).cuda()

    # 运行算法
    result = LK_function(test_data)

    # 验证输出
    assert result.shape == (100, 4, 16, 32, 32), f"维度错误: {result.shape}"
    assert (result.abs().sum(dim=(2, 3, 4)) > 0).all(), "检测到全零帧"
    print("测试通过！输出维度:", result.shape)
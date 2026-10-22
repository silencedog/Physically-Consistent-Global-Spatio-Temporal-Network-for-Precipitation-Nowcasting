from torch import nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from math import exp
from pytorch_msssim import MS_SSIM, SSIM
import utils.transform as transform

def weighted_l1_loss(pred: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    points = torch.tensor([10.0, 20.0, 30.0, 40.0])
    points = transform.minmax_norm(points)
    weight = (truth < points[0]) * 1 \
        + (torch.logical_and(truth >= points[0], truth < points[1])) * 2 \
        + (torch.logical_and(truth >= points[1], truth < points[2])) * 5 \
        + (torch.logical_and(truth >= points[2], truth < points[3])) * 10 \
        + (truth >= points[3]) * 30
    return torch.mean(weight * torch.abs(pred - truth))


def svre_loss(pred: torch.Tensor, truth: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    pred_cv = torch.std(pred, dim=(2, 3, 4)) / (torch.mean(pred, dim=(2, 3, 4)) + eps)
    truth_cv = torch.std(truth, dim=(2, 3, 4)) / (torch.mean(truth, dim=(2, 3, 4)) + eps)
    return F.l1_loss(pred_cv, truth_cv)


def d_loss(fake_score: torch.Tensor, real_score: torch.Tensor,
           loss_func: nn.Module = nn.MSELoss()) -> torch.Tensor:
    label = torch.ones_like(fake_score).type_as(fake_score)
    loss_pred = loss_func(fake_score, label * 0.0)
    loss_truth = loss_func(real_score, label * 1.0)
    d_loss = loss_pred + loss_truth
    return d_loss


def g_loss(fake_score: torch.Tensor, loss_func: nn.Module = nn.MSELoss()) -> torch.Tensor:
    label = torch.ones_like(fake_score).type_as(fake_score)
    g_loss = loss_func(fake_score, label * 1.0)
    return g_loss


class SSIMLoss(nn.Module):
    def __init__(self, convert_range: bool = False, **kwargs):
        """
        SSIM Loss, optionally converting input range from [-1,1] to [0,1]

        Args:
            convert_range:
            **kwargs:
        """
        super(SSIMLoss, self).__init__()
        self.convert_range = convert_range
        self.ssim_module = SSIM(data_range=1, size_average=True, channel=16)

    def forward(self, x, y):
        if self.convert_range:
            x = torch.div(torch.add(x, 1), 2)
            y = torch.div(torch.add(y, 1), 2)
        return 1.0 - self.ssim_module(x, y)


class MS_SSIMLoss(nn.Module):
    def __init__(self, convert_range: bool = False, **kwargs):
        """
        Multi-Scale SSIM Loss, optionally converting input range from [-1,1] to [0,1]

        Args:
            convert_range:
            **kwargs:
        """
        super(MS_SSIMLoss, self).__init__()
        self.convert_range = convert_range
        self.ssim_module = MS_SSIM(**kwargs)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        if self.convert_range:
            x = torch.div(torch.add(x, 1), 2)
            y = torch.div(torch.add(y, 1), 2)
        return 1.0 - self.ssim_module(x, y)



class Loss(nn.Module):
    def __init__(self, **kwargs):
        super(Loss, self).__init__()
        self.MSE_criterion = nn.MSELoss(reduction='mean')
        self.MAE_criterion = nn.L1Loss(reduction='mean')
        self.BCE_criterion = nn.BCELoss(reduction='mean')
        # self.SSIM = SSIM(data_range=1, size_average=True, channel=16)  # channel=1 for grayscale images
        # self.SSIM = SSIMLoss()
        # self.SSIM = SSIMLoss
    def forward(self, x_list, label):
        """B, T, C, H, W = label.shape
        ssim_loss = []
        for j in range(len(x_list)):
            ssim_loss_value = 0
            for i in range(T):
                # 假设 self.SSIM 计算并返回 1 - SSIM 作为损失
                ssim_loss_value += self.SSIM(x_list[j][:, i, :, :, :], label[:, i, :, :, :])

            # 对每个时序进行平均
            ssim_loss_value /= T
            ssim_loss.append(ssim_loss_value)

        total_ssim_loss = sum(ssim_loss) / len(x_list)"""

        loss = 2.0 * (self.MSE_criterion(x_list[0], label) + self.MAE_criterion(x_list[0], label))
        # loss = 2.0 * (5 * self.MSE_criterion(x_list[0], label) + self.MAE_criterion(x_list[0], label))
        return loss
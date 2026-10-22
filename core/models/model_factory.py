import os
import torch
import torch.nn as nn
from torch.nn import utils
from torch.optim import Adam
from core.models import network
from core.models.loss import Loss
import scipy.io as io
import numpy as np
from thop import profile

class Model(object):
    def __init__(self, configs):
        self.configs = configs
        self.total_length = configs.total_length
        self.input_length = configs.input_length
        # 分布式训练
        self.local_rank = configs.local_rank
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device("cuda", self.local_rank)

        self.Network = network.Network
        self.network = self.Network(configs).to(self.device)
        # 分布式训练
        self.network = nn.parallel.DistributedDataParallel(self.network, device_ids=[self.local_rank],
                                                           output_device=0, find_unused_parameters=False).to(self.device)
        self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network).to(self.device)
        self.optimizer = Adam(self.network.parameters(), lr=configs.lr)
        self.loss = Loss().to(self.device)

        dummy_frames = torch.randn(1, self.configs.total_length, self.configs.patch_size ** 2 * self.configs.img_channel,
                                   self.configs.img_height // self.configs.patch_size, self.configs.img_width // self.configs.patch_size).to(
            self.device)
        dummy_mask = torch.randn(1, self.configs.total_length - self.configs.input_length - 1, self.configs.patch_size ** 2 * self.configs.img_channel,
                                 self.configs.img_height // self.configs.patch_size, self.configs.img_width // self.configs.patch_size).to(self.device)

        # 注意：你的 network 可能还需要 batch_size 参数，因为调用是 self.network(frames_tensor, mask_tensor, batch_size)
        # 查看 network 的定义，如果第三个参数是 batch_size，则传入 dummy_frames.shape[0]
        flops, params = profile(self.network, inputs=(dummy_frames, dummy_mask, 1))
        print(f"Model FLOPs: {flops / 1e9:.2f} GFLOPs, Params: {params / 1e6:.2f} M")

    def save(self, ite=None):
        stats = {}
        stats['net_param'] = self.network.module.state_dict()
        if ite == None:
            checkpoint_path = os.path.join(self.configs.save_dir, 'radar_model.ckpt')
        else:
            checkpoint_path = os.path.join(self.configs.save_dir, 'radar_model_'+str(ite)+'.ckpt')
        # 分布式训练
        if torch.distributed.get_rank() == 0:
            torch.save(stats, checkpoint_path)
            print("save model to %s" % checkpoint_path)

    def load(self, saved_model):
        checkpoint_path = os.path.join(saved_model)
        # 分布式训练加载模型时出现多进程在GPU0上占用过多显存的问题解决方式(map_location='cpu')
        stats = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        # self.network.module.load_state_dict(stats['net_param'])
        self.network.module.load_state_dict(stats['net_param'], strict=False)
        print('model has been loaded in', checkpoint_path)

    def train(self, frames, mask):
        self.network.train()
        # frames_tensor.shape=[35, 15, 32, 32, 16]
        frames_tensor = torch.FloatTensor(frames).to(self.device)
        # frames_extra_tensor = torch.FloatTensor(frames_extra).to(self.device)
        mask_tensor = torch.FloatTensor(mask).to(self.device)
        # frames_tensor.shape=[35, 15, 16, 32, 32]
        frames_tensor = frames_tensor.permute((0, 1, 4, 2, 3)).contiguous()
        # frames_extra_tensor = frames_extra_tensor.permute(0, 1, 4, 2, 3).contiguous()
        # mask_tensor.shape=[35, 9, 16, 32, 32]
        mask_tensor = mask_tensor.permute((0, 1, 4, 2, 3)).contiguous()
        self.optimizer.zero_grad()
        # next_frames.shape=[35, 14, 16, 32, 32]
        next_frames = self.network(frames_tensor, mask_tensor, self.configs.train_batch_size)
        # loss = self.loss(next_frames, frames_tensor[:, 1:])
        loss = self.loss(next_frames, frames_tensor[:, 1:self.total_length])
        loss.backward()
        utils.clip_grad_norm_(self.network.parameters(), 0.50)
        # 找寻不能被求loss的异常变量
        for name, param in self.network.named_parameters():
            if param.grad is None:
                print(name)
        self.optimizer.step()
        return loss.detach().cpu().numpy()

    def valid(self, frames, mask):
        self.network.eval()
        frames_tensor = torch.FloatTensor(frames).to(self.device)
        # frames_extra_tensor = torch.FloatTensor(frames_extra).to(self.device)
        mask_tensor = torch.FloatTensor(mask).to(self.device)
        frames_tensor = frames_tensor.permute(0, 1, 4, 2, 3).contiguous()
        # frames_extra_tensor = frames_extra_tensor.permute(0, 1, 4, 2, 3).contiguous()
        mask_tensor = mask_tensor.permute(0, 1, 4, 2, 3).contiguous()
        next_frames = self.network(frames_tensor, mask_tensor, self.configs.valid_batch_size)
        self.network.train()
        return next_frames[0].detach().cpu().numpy()

    def test(self, frames, mask):
        frames_tensor = torch.FloatTensor(frames).to(self.device)
        # frames_extra_tensor = torch.FloatTensor(frames_extra).to(self.device)
        mask_tensor = torch.FloatTensor(mask).to(self.device)
        frames_tensor = frames_tensor.permute(0, 1, 4, 2, 3).contiguous()
        # frames_extra_tensor = frames_extra_tensor.permute(0, 1, 4, 2, 3).contiguous()
        mask_tensor = mask_tensor.permute(0, 1, 4, 2, 3).contiguous()
        self.network.eval()
        next_frames = self.network(frames_tensor, mask_tensor, self.configs.test_batch_size)
        return next_frames[0].detach().cpu().numpy()





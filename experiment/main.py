import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from CIKM import *
from slover import *
import random
import numpy as np
import torch
import argparse
from data.CIKM.cikm_radar import Radar
from torch.utils.data import DataLoader
from core.models.model_factory import Model
from torch.utils.data.distributed import DistributedSampler

# 分布式训练
torch.distributed.init_process_group(backend="nccl")

def parse_args():

    parser = argparse.ArgumentParser(description='Precipitation Nowcasting')

    # training: 1 / test: 0
    parser.add_argument('--is_training', type=int, default=0)

    # offline: 1 / online: 0
    parser.add_argument('--is_offline', type=int, default=1)

    # test selection
    parser.add_argument('--result_model_name', type=str, default='run-10')
    parser.add_argument('--result_num_star', type=str, default=35)
    parser.add_argument('--result_num_end', type=str, default=43)

    # 分布式训练
    parser.add_argument('--local_rank', type=int, default=torch.distributed.get_rank())

    # data
    parser.add_argument('--dataset_name', type=str, default='CIKM')  # NMIC
    parser.add_argument('--dataset_type', type=str, default='radar')
    parser.add_argument('--data_root', type=str, default='/code/data')  # /home/code/data
    parser.add_argument('--save_dir', type=str, default='/code/6/model_lib')
    parser.add_argument('--gen_frm_dir', type=str, default='/code/6/results/')
    parser.add_argument('--total_length', type=int, default=15)  # 18
    parser.add_argument('--input_length', type=int, default=5)  # 6
    parser.add_argument('--infer_length', type=int, default=10)
    parser.add_argument('--img_width', type=int, default=128)
    parser.add_argument('--img_height', type=int, default=128)
    parser.add_argument('--img_channel', type=int, default=1)

    # model
    parser.add_argument('--model_name', type=str, default='network')
    parser.add_argument('--PySetps', type=bool,default=False)
    parser.add_argument('--pretrained_model', type=str, default='')  # /home/code/fifthwork/model_lib/run-7/radar_model_2.ckpt
    parser.add_argument('--result_model_dir', type=str, default='')
    parser.add_argument('--output_dir', type=str, default='')
    parser.add_argument('--multi_length', type=str, default=3)
    parser.add_argument('--num_layers', type=str, default=4)
    parser.add_argument('--num_hidden', type=str, default=64)
    parser.add_argument('--filter_size', type=int, default=5)
    parser.add_argument('--num_layer', type=int, default=1)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--d_model', type=int, default=16)
    parser.add_argument('--n_encoder_layers', type=int, default=3)
    parser.add_argument('--n_decoder_layers', type=int, default=3)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--patch_size', type=int, default=4)  # 4
    parser.add_argument('--patch_size_list', type=int, default=[4])  # 4
    parser.add_argument('--layer_norm', type=int, default=1)

    # scheduled sampling
    parser.add_argument('--scheduled_sampling', type=int, default=1)
    parser.add_argument('--sampling_stop_iter', type=int, default=500)  # 500
    parser.add_argument('--sampling_start_value', type=float, default=1.0)
    parser.add_argument('--sampling_changing_rate', type=float, default=0.00002)

    # optimization
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--reverse_input', type=int, default=1)
    parser.add_argument('--train_batch_size', type=int, default=12)
    parser.add_argument('--valid_batch_size', type=int, default=100)
    parser.add_argument('--test_batch_size', type=int, default=20)
    parser.add_argument('--max_iterations', type=int, default=200)  # 80000
    parser.add_argument('--display_interval', type=int, default=1)
    parser.add_argument('--test_interval', type=int, default=1)
    parser.add_argument('--snapshot_interval', type=int, default=5)
    parser.add_argument('--num_save_samples', type=int, default=10)
    parser.add_argument('--limit', type=int, default=1000)
    parser.add_argument('--n_gpu', type=int, default=1)
    return parser.parse_args()

def init_seeds(seed, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:  # faster, less reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
#  validation
def main(args):
    # init_seeds(1 + args.local_rank)
    init_seeds(1)
    train_data = Radar(
        data_type='train',
        data_root="%s/%s" % (args.data_root, args.dataset_name)
    )
    valid_data = Radar(
        data_type='validation',
        data_root="%s/%s" % (args.data_root, args.dataset_name)
    )
    test_data = Radar(
        data_type='test',
        data_root="%s/%s" % (args.data_root, args.dataset_name)
    )
    # 分布式训练
    train_sampler = DistributedSampler(train_data)
    # valid_sampler = DistributedSampler(valid_data)
    # test_sampler = DistributedSampler(test_data)

    train_loader = DataLoader(train_data,
                              num_workers=4,
                              batch_size=args.train_batch_size,
                              shuffle=False,
                              sampler=train_sampler,
                              drop_last=True,
                              pin_memory=True)
    valid_loader = DataLoader(valid_data,
                              num_workers=4,
                              batch_size=args.valid_batch_size,
                              shuffle=False,
                              # sampler=valid_sampler,
                              drop_last=True,
                              pin_memory=True)
    test_loader = DataLoader(test_data,
                             num_workers=4,
                             batch_size=args.test_batch_size,
                             shuffle=False,
                             # sampler=test_sampler,
                             drop_last=False,
                             pin_memory=False)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if not os.path.exists(args.gen_frm_dir):
        os.makedirs(args.gen_frm_dir)
    model = Model(args)
    if args.is_training:
        # Generate a folder for saving results
        if torch.distributed.get_rank() == 0:
            run = 0
            # Training
            while os.path.exists("%s/run-%d" % (args.save_dir, run)):
                run += 1
            os.mkdir("%s/run-%d" % (args.save_dir, run))
            args.save_dir = "%s/run-%d" % (args.save_dir, run)
        if args.dataset_name == 'CIKM':
            wrapper_train(model, train_loader, valid_loader, padding_CIKM_data, schedule_sampling, args)
        else:
            wrapper_train(model, train_loader, valid_loader, padding_data, schedule_sampling, args)
    else:
        if torch.distributed.get_rank() == 0:
            result_num = 0
            for i in range(args.result_num_star, args.result_num_end):
                while os.path.exists("%s/result-%d" % (args.gen_frm_dir, result_num)):
                    result_num += 1
                os.mkdir("%s/result-%d" % (args.gen_frm_dir, result_num))
                args.output_dir = "%s/result-%d" % (args.gen_frm_dir, result_num)
                args.result_model_dir = "%s/%s/radar_model_%d.ckpt" % (args.save_dir, args.result_model_name, i)
                if args.dataset_name == 'CIKM':
                    wrapper_test(model, padding_CIKM_data, unpadding_CIKM_data, test_loader, args, is_save=True)
                else:
                    wrapper_test(model, padding_data, unpadding_data, test_loader, args, is_save=True)


"""def setup():
    torch.distributed.init_process_group("nccl")

def cleanup():
    torch.distributed.destroy_process_group()"""

if __name__ == "__main__":

    args = parse_args()
    main(args)




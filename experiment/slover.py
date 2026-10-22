import os
import cv2
import math
import torch
import time
import scipy.io as io
import numpy as np
from utils import preprocess
from pytorch_msssim import SSIM

def wrapper_test(model, padding_data, unpadding_data, test_loader, args, is_save=True):
    model.load(args.result_model_dir)
    # test_sample_root = "%s/%s/%s/" % (args.data_root, args.dataset_name, 'test')
    test_save_root = args.output_dir
    if not os.path.exists(test_save_root):
        os.mkdir(test_save_root)
    test_output = open("%s/result.txt" % args.gen_frm_dir, 'w')
    avg_mae = 0
    avg_mse = 0
    count = 1
    # eta = args.sampling_start_value
    real_input_flag = np.zeros(
        (args.test_batch_size,
         args.total_length - args.input_length - 1,
         args.img_width // args.patch_size,
         args.img_width // args.patch_size,
         args.patch_size ** 2 * args.img_channel))
    # 10 = 15 - 5
    output_length = args.infer_length

    with torch.no_grad():
        for i_batch, batch_data in enumerate(test_loader):
            # ims.shape=tuple(8, 15, 101, 101, 1)
            ims = batch_data[0].numpy()
            # label tars.shape=(8, 10, 101, 101, 1)
            tars = ims[:, -output_length:]
            # 文件夹名字
            cur_fold = batch_data[1]

            # sample_fold = test_loader.sampler.data_source.data_root
            # sample_fold_type = test_loader.sampler.data_source.data_type
            if args.PySetps:
                """ims_extra = ims[:, 0:args.input_length]
                ims_extra = extrapolation(ims_extra, args, args.test_batch_size)
                ims_extra = padding_data(ims_extra, args)
                ims_extra = preprocess.reshape_patch(ims_extra, args.patch_size)"""
            else:
                ims_extra = ims
            # ims.shape=tuple(8, 15, 128, 128, 1) 填充
            ims = padding_data(ims, args)
            # ims.shape=tuple(8, 15, 32, 32, 16) 分割
            ims = preprocess.reshape_patch(ims, args.patch_size)
            # cloud_shift = np.array(cloud_shift[0, 0, :, :, :].cpu())
            # io.savemat('cloud_shift.mat', {'cloud_shift': cloud_shift})
            # img_gen.shape=ndarray(8, 14, 16, 32, 32)
            img_gen = model.test(ims, real_input_flag)
            # img_gen.shape=ndarray(8, 14, 32, 32, 16)
            img_gen = img_gen.transpose((0, 1, 3, 4, 2))
            # img_gen.shape=ndarray(8, 14, 128, 128, 1)
            img_gen = preprocess.reshape_patch_back(img_gen, args.patch_size)
            # KL = preprocess.reshape_patch_back(KL, args.patch_size)
            # KL = np.array(KL)
            # KL = np.squeeze(KL)
            # io.savemat('/home/code/PredRANN-master/KL.mat', {'KL': KL})
            # img_out.shape=ndarray(8, 10, 101, 101, 1)
            img_out = unpadding_data(img_gen[:, -output_length:])
            # Calculate MSE
            mse = np.mean(np.square(img_out - tars))
            print('--- Current MSE: %.10f ---' % mse)
            avg_mse += mse
            # Calculate MAE
            mae = np.mean(np.abs(img_out-tars))
            print('--- Current MAE: %.10f ---' % mae)
            if torch.distributed.get_rank() == 0:
                print('--- Current MAE: %.10f ---' % mae, file=test_output)
            avg_mae += mae
            img_out[img_out < 0] = 0
            img_out[img_out > 1] = 1
            img_out = (img_out*255.0).astype(np.uint8)
            # count: image_bag / batch_size
            count = count + 1
            if is_save and torch.distributed.get_rank() == 0:
                for bat_ind in range(args.test_batch_size):
                    cur_batch_data = img_out[bat_ind, :, :, :, 0]
                    cur_sample_fold = os.path.join(test_save_root, cur_fold[bat_ind])
                    if not os.path.exists(cur_sample_fold):
                        os.mkdir(cur_sample_fold)
                    for t in range(args.infer_length):
                        cur_save_path = os.path.join(cur_sample_fold, 'img_'+str(t+args.input_length+1)+'.png')
                        # cur_save_path = os.path.join(cur_sample_fold, str(test_sample_fold[t + args.input_length]))
                        cur_img = cur_batch_data[t]
                        cv2.imwrite(cur_save_path, cur_img)
    if torch.distributed.get_rank() == 0:
        avg_mse_value = str(avg_mse / count)
        print('Finally MSE:', avg_mse_value)
        print('--- Finally MSE: %.10f ---' % float(avg_mse_value), file=test_output)

        avg_mae_value = str(avg_mae / count)
        print('Finally MAE:', avg_mae_value)
        print('--- Finally MAE: %.10f ---' % float(avg_mae_value), file=test_output)

    return mse / count

def wrapper_valid(model, padding_data, valid_loader, args):
    output_length = args.total_length - args.input_length
    mse = 0
    ssim_value = 0
    # ssim = SSIM(data_range=1, size_average=True, channel=args.patch_size ** 2)
    real_input_flag = np.zeros(
        (args.valid_batch_size,
         args.total_length - args.input_length - 1,
         args.img_width // args.patch_size,
         args.img_width // args.patch_size,
         args.patch_size ** 2 * args.img_channel))
    with torch.no_grad():
        # start = time.time()
        for i_batch, batch_data in enumerate(valid_loader):
            ims = batch_data.numpy()
            if args.PySetps:
                """ims_extra = ims[:, 0:args.input_length]
                ims_extra = extrapolation(ims_extra, args, args.valid_batch_size)
                ims_extra = padding_data(ims_extra, args)
                ims_extra = preprocess.reshape_patch(ims_extra, args.patch_size)"""
            else:
                ims_extra = ims
            ims = padding_data(ims, args)
            ims = preprocess.reshape_patch(ims, args.patch_size)
            img_out = model.valid(ims, real_input_flag)
            tars = torch.FloatTensor(ims[:, -output_length:])
            tars = tars.permute(0, 1, 4, 2, 3).contiguous()
            img_out = torch.FloatTensor(img_out[:, -output_length:])
            mse += torch.mean(torch.square(img_out - tars))
            """for i in range(output_length):
                ssim_value += ssim(img_out[:, i], tars[:, i])"""
            ssim_value = 0
        # ssim_value = ssim_value / args.valid_batch_size
        # end = time.time()
        # end = end - start
    return mse/(i_batch+1), ssim_value / ((i_batch+1) * output_length)

def wrapper_train(model, train_loader, valid_loader, padding_data, schedule_sampling, args):
    if args.pretrained_model:
        model.load(args.pretrained_model)
    log_output = open("%s/log.txt" % args.save_dir, 'w')
    eta = args.sampling_start_value
    best_mse = math.inf
    best_ssim = 0
    ssim_index = 0
    mse_index = 0
    tolerate = 0
    iter_num = len(train_loader.dataset) / (args.n_gpu * args.train_batch_size)

    for itr in range(0, args.max_iterations):
        # 分布式训练
        # 设置sampler的epoch，
        # DistributedSampler需要这个来指定shuffle方式，
        # 通过维持各个进程之间的相同随机数种子使不同进程能获得同样的shuffle效果。
        train_loader.sampler.set_epoch(itr)
        for i_batch, batch_data in enumerate(train_loader):
            # ims.shape=[35, 15, 101, 101, 1]
            ims = batch_data.numpy()
            # ims.shape=[35, 5, 101, 101, 1]
            # ims.shape=[35, 15, 128, 128, 1]
            if args.PySetps:
                """ims_extra = ims[:, 0:args.input_length]
                ims_extra = extrapolation(ims_extra, args, args.train_batch_size)
                ims_extra = padding_data(ims_extra, args)
                ims_extra = preprocess.reshape_patch(ims_extra, args.patch_size)"""
            else:
                ims_extra = ims
            ims = padding_data(ims, args)
            # ims.shape=[35, 15, 32, 32, 16]
            ims = preprocess.reshape_patch(ims, args.patch_size)
            eta, real_input_flag = schedule_sampling(eta, itr, args)
            cost = model.train(ims, real_input_flag)
            if args.reverse_input:
                ims_rev = np.flip(ims, axis=1).copy()
                cost += model.train(ims_rev, real_input_flag)
                cost = cost / 2
            if (i_batch+1) % args.display_interval == 0:
                print('Epoch: [%d/%d], Iteration: [%d/%d], Training loss: %.20f, lr: %s' %
                      (itr, args.max_iterations, i_batch, iter_num, cost, args.lr))
                print('Epoch: [%d/%d], Iteration: [%d/%d], Training loss: %.20f, lr: %s' %
                      (itr, args.max_iterations, i_batch, iter_num, cost, args.lr), file=log_output)
        if (itr+1) % args.test_interval == 0 and torch.distributed.get_rank() == 0:
            print('Validation')
            """curr_mse, curr_ssim = wrapper_valid(model, padding_data, valid_loader, args)
            print('--- Save File: %s, Best MSE: %.10f, Curr MSE: %.10f, Best Index: %d---' % (args.save_dir.split('/')[-1], best_mse, curr_mse, mse_index))
            print('--- Save File: %s, Best SSIM: %.10f, Curr SSIM: %.10f, Best Index: %d ---' % (args.save_dir.split('/')[-1], best_ssim, curr_ssim, ssim_index))
            print('--- Save File: %s, Best MSE: %.10f, Curr MSE: %.10f, Best Index: %d ---' % (args.save_dir.split('/')[-1], best_mse, curr_mse, mse_index), file=log_output)
            print('--- Save File: %s, Best SSIM: %.10f, Curr SSIM: %.10f, Best Index: %d ---' % (args.save_dir.split('/')[-1], best_ssim, curr_ssim, ssim_index), file=log_output)

            if best_ssim < curr_ssim:
                best_ssim = curr_ssim
                ssim_index = itr

            if curr_mse < best_mse:
                best_mse = curr_mse
                tolerate = 0
                mse_index = itr
            else:
                tolerate = tolerate+1"""

            model.save(itr)
            # Stop automatically when meeting the best result
            """if tolerate == args.limit:
                print('The Best MSE is: ', str(best_mse))
                break"""


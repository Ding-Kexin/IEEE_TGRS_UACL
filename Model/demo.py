# -*- coding: utf-8 -*-
"""
@Time ： 2023/3/6 13:26
@Author ：Kexin Ding
@FileName ：demo.py
"""
import torch
import argparse
import torch.utils.data as Data
from scipy.io import loadmat
import numpy as np
import time
import os
from utils import train_patch, setup_seed, print_args, show_calaError
import sys
from UACL import train_network
# -------------------------------------------------------------------------------
# Parameter Setting
parser = argparse.ArgumentParser("UACL")
parser.add_argument('--gpu_id', default='3', help='gpu id')
parser.add_argument('--seed', type=int, default=0, help='number of seed')
parser.add_argument('--epoches', type=int, default=300, help='epoch number')
parser.add_argument('--learning_rate', type=float, default=5e-4, help='learning rate')
parser.add_argument('--factor_lambda', type=float, default=0.1, help='theta')
parser.add_argument('--dataset', choices=['Muufl', 'Trento', 'Houston', 'Augsburg'], default='Muufl', help='dataset')
parser.add_argument('--num_class', choices=[11, 6, 15, 7], default=11, help='number of classes')
parser.add_argument('--batch_size', type=int, default=64, help='number of batch size')
parser.add_argument('--patch_size', type=int, default=16, help='number1 of patches')
parser.add_argument('--num_labelled', type=int, default=220, help='number of sampling from unlabeled samples')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def train_1time():
    # -------------------------------------------------------------------------------
    # prepare data
    num_train_1 = [20]
    k = 0
    number = 4
    if args.dataset == "Houston":
        DataPath1 = r'/home/server04/dkx/dkx_experiment/dataset/houston/Houston.mat'
        DataPath2 = r'/home/server04/dkx/dkx_experiment/dataset/houston/LiDAR.mat'
        Data1 = loadmat(DataPath1)['img']  # (349,1905,144)
        Data2 = loadmat(DataPath2)['img']
        LabelPath = r'/home/server04/dkx/dkx_experiment/dataset/houston/train_test/%d/train_test_gt_%d.mat' % (
            num_train_1[k], number)
    elif args.dataset == "Muufl":
        DataPath1 = r'/home/server04/dkx/dkx_experiment/dataset/Muufl/hsi.mat'
        DataPath2 = r'/home/server04/dkx/dkx_experiment/dataset/Muufl/lidar_DEM.mat'
        Data1 = loadmat(DataPath1)['hsi']
        Data2 = loadmat(DataPath2)['lidar']
        LabelPath = r'/home/server04/dkx/dkx_experiment/dataset/Muufl/train_test/%d/train_test_gt_%d.mat' % (
            num_train_1[k], number)
    Data1 = Data1.astype(np.float32)
    Data2 = Data2.astype(np.float32)
    [m1, n1, l1] = np.shape(Data1)
    Data2 = Data2.reshape([m1, n1, -1])  # when lidar is one band, this is used
    height1, width1, band1 = Data1.shape
    height2, width2, band2 = Data2.shape
    # data size
    print("height1={0},width1={1},band1={2}".format(height1, width1, band1))
    print("height2={0},width2={1},band2={2}".format(height2, width2, band2))
    TrLabel = loadmat(LabelPath)['train_data']
    TsLabel = loadmat(LabelPath)['test_data']
    patchsize = args.patch_size  # input spatial size for 2D-CNN
    pad_width = np.floor(patchsize / 2)
    pad_width = int(pad_width)  # 8
    TrainPatch1, TrainPatch2, TrainLabel = train_patch(Data1, Data2, patchsize, pad_width, TrLabel)
    TestPatch1, TestPatch2, TestLabel = train_patch(Data1, Data2, patchsize, pad_width, TsLabel)
    train_dataset = Data.TensorDataset(TrainPatch1, TrainPatch2, TrainLabel)
    train_loader = Data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    print('Data1 Training size and testing size are:', TrainPatch1.shape, 'and', TestPatch1.shape)
    print('Data2 Training size and testing size are:', TrainPatch2.shape, 'and', TestPatch2.shape)
    tic1 = time.time()
    pred_y, val_acc = train_network(train_loader, TestPatch1, TestPatch2, TestLabel,
                                    LR=args.learning_rate,
                                    EPOCH=args.epoches, l1=band1, l2=band2,
                                    Classes=args.num_class, num_train=num_train_1[k], order=number,
                                    patch_size=args.patch_size, num_labelled=args.num_labelled, factor_lambda=args.factor_lambda)
    pred_y.type(torch.FloatTensor)
    TestLabel.type(torch.FloatTensor)
    print("***********************Train raw***************************")
    print("Maxmial Accuracy: %f, index: %i" % (max(val_acc), val_acc.index(max(val_acc))))
    toc1 = time.time()
    time_1 = toc1 - tic1
    print('1st training complete in {:.0f}m {:.0f}s'.format(time_1 / 60, time_1 % 60))
    OA, Kappa, CA, AA = show_calaError(pred_y, TestLabel)
    toc = time.time()
    time_all = toc - tic1
    print('All process complete in {:.0f}m {:.0f}s'.format(time_all / 60, time_all % 60))
    print("**************************************************")
    print("Parameter:")
    print_args(vars(args))


if __name__ == '__main__':
    setup_seed(args.seed)
    train_1time()

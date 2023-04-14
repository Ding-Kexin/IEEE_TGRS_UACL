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
parser.add_argument('--dataset', choices=['Muufl', 'Trento', 'Houston', 'Augsburg'], default='Houston', help='dataset')
parser.add_argument('--num_class', choices=[11, 6, 15, 7], default=15, help='number of classes')
parser.add_argument('--batch_size', type=int, default=64, help='number of batch size')
parser.add_argument('--patch_size', type=int, default=16, help='number1 of patches')
parser.add_argument('--num_labelled', type=int, default=300, help='number of sampling from unlabeled samples')
parser.add_argument('--training_mode', choices=['one_time', 'ten_times', 'test_all', 'train_standard'], default='one_time', help='training times')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
def text_create(name, num_train_1):
  desktop_path = r"/home/server04/dkx/dkx_experiment/SSCL_Net/code/SSCL_houston/records/%d//"%(num_train_1)
  # 新创建的txt文件的存放路径
  full_path = desktop_path + name + '.txt' # 也可以创建一个.doc的word文档
  file = open(full_path, 'w')

def train_1time():  # 0.955462
    # setup_seed(args.seed)
    # -------------------------------------------------------------------------------
    # prepare data
    DataPath1 = r'/home/server04/dkx/dkx_experiment/dataset/houston/Houston.mat'
    DataPath2 = r'/home/server04/dkx/dkx_experiment/dataset/houston/LiDAR.mat'
    Data1 = loadmat(DataPath1)['img']  # (349,1905,144)
    Data2 = loadmat(DataPath2)['img']
    Data1 = Data1.astype(np.float32)
    Data2 = Data2.astype(np.float32)
    [m1, n1, l1] = np.shape(Data1)
    Data2 = Data2.reshape([m1, n1, -1])  # when lidar is one band, this is used
    height1, width1, band1 = Data1.shape
    height2, width2, band2 = Data2.shape
    # data size
    print("height1={0},width1={1},band1={2}".format(height1, width1, band1))
    print("height2={0},width2={1},band2={2}".format(height2, width2, band2))
    num_train_1 = [20]
    k = 0
    number = 4
    LabelPath_10TIMES = r'/home/server04/dkx/dkx_experiment/dataset/houston/train_test/%d/train_test_gt_%d.mat' % (
        num_train_1[k], number)
    TrLabel_10TIMES = loadmat(LabelPath_10TIMES)['train_data']
    TsLabel_10TIMES = loadmat(LabelPath_10TIMES)['test_data']
    patchsize = args.patch_size  # input spatial size for 2D-CNN
    pad_width = np.floor(patchsize / 2)
    pad_width = int(pad_width)  # 8
    TrainPatch1, TrainPatch2, TrainLabel = train_patch(Data1, Data2, patchsize, pad_width, TrLabel_10TIMES)
    TestPatch1, TestPatch2, TestLabel = train_patch(Data1, Data2, patchsize, pad_width, TsLabel_10TIMES)
    train_dataset = Data.TensorDataset(TrainPatch1, TrainPatch2, TrainLabel)
    train_loader = Data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    # filename = 'SSCL_baseline_theta0.1_seed0_%d_%d' % (num_train_1[k], number)
    # text_create(filename, num_train_1[k])
    # output = sys.stdout
    # outputfile = open(r"/home/server04/dkx/dkx_experiment/SSCL_Net/code/SSCL_houston/records/%d//" % (
    #     num_train_1[k]) + filename + '.txt', 'a')
    # sys.stdout = outputfile
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
    # outputfile.close()


def train_10times():
    # setup_seed(args.seed)
    # -------------------------------------------------------------------------------
    # prepare data
    DataPath1 = r'/home/server04/dkx/dkx_experiment/dataset/houston/Houston.mat'
    DataPath2 = r'/home/server04/dkx/dkx_experiment/dataset/houston/LiDAR.mat'
    Data1 = loadmat(DataPath1)['img']  # (349,1905,144)
    Data2 = loadmat(DataPath2)['img']
    Data1 = Data1.astype(np.float32)
    Data2 = Data2.astype(np.float32)
    [m1, n1, l1] = np.shape(Data1)
    Data2 = Data2.reshape([m1, n1, -1])  # when lidar is one band, this is used
    height1, width1, band1 = Data1.shape
    height2, width2, band2 = Data2.shape
    # data size
    print("height1={0},width1={1},band1={2}".format(height1, width1, band1))
    print("height2={0},width2={1},band2={2}".format(height2, width2, band2))
    num_train_1 = [100]
    for k in range(len(num_train_1)):
        OA_ALL = []
        AA_ALL = []
        CA_ALL = []
        KAPPA_ALL = []
        for number in range(10, 0, -1):
            LabelPath_10TIMES = r'/home/server04/dkx/dkx_experiment/dataset/houston/train_test/%d/train_test_gt_%d.mat' % (
            num_train_1[k], number)  # OA:50/records = 0.9686(50)/0.8995(records)/0.9238(records pre_train)
            TrLabel_10TIMES = loadmat(LabelPath_10TIMES)['train_data']
            TsLabel_10TIMES = loadmat(LabelPath_10TIMES)['test_data']
            patchsize = args.patch_size  # input spatial size for 2D-CNN
            pad_width = np.floor(patchsize / 2)
            pad_width = int(pad_width)  # 8
            TrainPatch1, TrainPatch2, TrainLabel = train_patch(Data1, Data2, patchsize, pad_width, TrLabel_10TIMES)
            TestPatch1, TestPatch2, TestLabel = train_patch(Data1, Data2, patchsize, pad_width, TsLabel_10TIMES)
            train_dataset = Data.TensorDataset(TrainPatch1, TrainPatch2, TrainLabel)
            train_loader = Data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
            filename = 'SSCL_baseline_theta0.1_seed0_%d_%d' % (num_train_1[k], number)
            text_create(filename, num_train_1[k])
            output = sys.stdout
            outputfile = open(r"/home/server04/dkx/dkx_experiment/SSCL_Net/code/SSCL_houston/records/%d//" % (
            num_train_1[k]) + filename + '.txt', 'a')
            sys.stdout = outputfile
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
            print("***********************Train and test result record***************************")
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
            outputfile.close()
            OA_ALL.append(OA)
            AA_ALL.append(AA)
            CA_ALL.append(CA)
            KAPPA_ALL.append(Kappa)
            del TestPatch1, TrainPatch1, TestPatch2, TrainPatch2, TrainLabel, TestLabel, train_dataset, pred_y, \
                val_acc, train_loader
        filename = 'SSCL_baseline_theta0.1_seed0_%d_mean' % (num_train_1[k])
        text_create(filename, num_train_1[k])
        output = sys.stdout
        outputfile = open(r"/home/server04/dkx/dkx_experiment/SSCL_Net/code/SSCL_houston/records/%d//" % (
            num_train_1[k]) + filename + '.txt', 'a')
        sys.stdout = outputfile
        print("***********************Train raw***************************")
        print('OA_10times:', OA_ALL)
        print('OA_10times_mean:', np.mean(OA_ALL))
        print('AA_10times:', AA_ALL)
        print('AA_10times_mean:', np.mean(AA_ALL))
        print('KAPPA_10times:', KAPPA_ALL)
        print('KAPPA_10times_mean:', np.mean(KAPPA_ALL))
        print(CA_ALL)
        outputfile.close()


if __name__ == '__main__':
    setup_seed(args.seed)
    if args.training_mode == 'one_time':
        train_1time()
    elif args.training_mode == 'ten_times':
        train_10times()

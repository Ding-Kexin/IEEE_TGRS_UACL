# -*- coding: utf-8 -*-
"""
@Time ： 2023/3/6 13:26
@Author ：Kexin Ding
@FileName ：utils.py
"""
import torch
import torch.backends.cudnn as cudnn
import numpy as np


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    cudnn.benchmark = False


def train_patch(Data1, Data2, patchsize, pad_width, Label):
    [m1, n1, l1] = np.shape(Data1)
    Data2 = Data2.reshape([m1, n1, -1])
    [m2, n2, l2] = np.shape(Data2)
    for i in range(l1):
        Data1[:, :, i] = (Data1[:, :, i] - Data1[:, :, i].min()) / (Data1[:, :, i].max() - Data1[:, :, i].min())
    x1 = Data1
    for i in range(l2):
        Data2[:, :, i] = (Data2[:, :, i] - Data2[:, :, i].min()) / (Data2[:, :, i].max() - Data2[:, :, i].min())
    x2 = Data2
    x1_pad = np.empty((m1 + patchsize, n1 + patchsize, l1), dtype='float32')
    x2_pad = np.empty((m2 + patchsize, n2 + patchsize, l2), dtype='float32')
    for i in range(l1):
        temp = x1[:, :, i]
        temp2 = np.pad(temp, pad_width, 'symmetric')
        x1_pad[:, :, i] = temp2
    for i in range(l2):
        temp = x2[:, :, i]
        temp2 = np.pad(temp, pad_width, 'symmetric')
        x2_pad[:, :, i] = temp2
    # construct the training and testing set
    [ind1, ind2] = np.where(Label > 0)
    TrainNum = len(ind1)
    TrainPatch1 = np.empty((TrainNum, l1, patchsize, patchsize), dtype='float32')
    TrainPatch2 = np.empty((TrainNum, l2, patchsize, patchsize), dtype='float32')
    TrainLabel = np.empty(TrainNum)
    ind3 = ind1 + pad_width
    ind4 = ind2 + pad_width
    for i in range(len(ind1)):
        patch1 = x1_pad[(ind3[i] - pad_width):(ind3[i] + pad_width), (ind4[i] - pad_width):(ind4[i] + pad_width), :]
        patch1 = np.transpose(patch1, (2, 0, 1))
        TrainPatch1[i, :, :, :] = patch1
        patch2 = x2_pad[(ind3[i] - pad_width):(ind3[i] + pad_width), (ind4[i] - pad_width):(ind4[i] + pad_width), :]
        patch2 = np.transpose(patch2, (2, 0, 1))
        TrainPatch2[i, :, :, :] = patch2
        patchlabel = Label[ind1[i], ind2[i]]
        TrainLabel[i] = patchlabel
    # change data to the input type of PyTorch
    TrainPatch1 = torch.from_numpy(TrainPatch1)
    TrainPatch2 = torch.from_numpy(TrainPatch2)
    TrainLabel = torch.from_numpy(TrainLabel) - 1
    TrainLabel = TrainLabel.long()
    return TrainPatch1, TrainPatch2, TrainLabel


def print_args(args):
    for k, v in zip(args.keys(), args.values()):
        print("{0}: {1}".format(k, v))


def CalAccuracy(predict,label):
    n = label.shape[0]
    OA = torch.sum(predict==label)*1.0/n
    correct_sum = torch.zeros((max(label)+1))
    reali = torch.zeros((max(label)+1))
    predicti = torch.zeros((max(label)+1))
    CA = torch.zeros((max(label)+1))
    for i in range(0, max(label) + 1):
        correct_sum[i] = torch.sum(label[np.where(predict == i)] == i)
        reali[i] = torch.sum(label == i)
        predicti[i] = torch.sum(predict == i)
        CA[i] = correct_sum[i] / reali[i]

    Kappa = (n * torch.sum(correct_sum) - torch.sum(reali * predicti)) * 1.0 / (n * n - torch.sum(reali * predicti))
    AA = torch.mean(CA)
    return OA, Kappa, CA, AA


def show_calaError(val_predict_labels, val_true_labels):
   val_predict_labels = torch.squeeze(val_predict_labels)
   val_true_labels = torch.squeeze(val_true_labels)
   OA, Kappa, CA, AA = CalAccuracy(val_predict_labels, val_true_labels)
   # ic(OA, Kappa, CA, AA)
   print("OA: %f, Kappa: %f,  AA: %f" % (OA, Kappa, AA))
   print("CA: ",)
   print(CA)
   return OA, Kappa, CA, AA


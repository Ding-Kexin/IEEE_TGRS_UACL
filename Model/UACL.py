# -*- coding: utf-8 -*-
"""
@Time ： 2023/3/6 13:26
@Author ：Kexin Ding
@FileName ：UACL.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import cycle
import torch.utils.data as Data


class Encoder(nn.Module):
    def __init__(self, l1, l2):
        super(Encoder, self).__init__()
        self.spa1 = nn.Sequential(
            nn.Conv2d(l1, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.spa2 = nn.Sequential(
            nn.Conv2d(l2, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.spa3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.spa4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),  # No effect on order
            nn.MaxPool2d(2),
        )
        self.spe1 = nn.Sequential(
            # nn.Linear(l1, 32),  # same as Conv1d, input(b, l)
            nn.Conv1d(l1, 32, 1),  # same as Linear, input(b, l, 1)
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.spe2 = nn.Sequential(
            nn.Conv1d(32, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.spe3 = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.h_factor = torch.nn.Parameter(torch.Tensor([0.5]))  # alpha
        self.l_factor = torch.nn.Parameter(torch.Tensor([0.5]))  # 1 - alpha

    def forward(self, h_spa, l_spa, h_spe):
        h_spa1 = self.spa1(h_spa)
        l_spa1 = self.spa2(l_spa)
        h_spe1 = self.spe1(h_spe)
        fused_spa1 = self.h_factor * h_spa1 + self.l_factor * l_spa1
        fused_ss1 = h_spe1.unsqueeze(-1) * fused_spa1

        h_spa2 = self.spa3(h_spa1)
        l_spa2 = self.spa3(l_spa1)
        h_spe2 = self.spe2(h_spe1)
        fused_spa2 = self.h_factor * h_spa2 + self.l_factor * l_spa2
        fused_ss2 = h_spe2.unsqueeze(-1) * fused_spa2

        h_spa3 = self.spa4(h_spa2)
        l_spa3 = self.spa4(l_spa2)
        h_spe3 = self.spe3(h_spe2)
        fused_spa3 = self.h_factor * h_spa3 + self.l_factor * l_spa3
        fused_ss3 = h_spe3.unsqueeze(-1) * fused_spa3

        return fused_ss1, fused_ss2, fused_ss3


class Classifier(nn.Module):
    def __init__(self, Classes):
        super(Classifier, self).__init__()
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(32, 16, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(16, 16, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.conv3_1 = nn.Sequential(
            nn.Conv2d(16, Classes, 1),
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(64, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(32, 16, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.conv3_2 = nn.Sequential(
            nn.Conv2d(16, Classes, 1),
        )
        self.conv1_3 = nn.Sequential(
            nn.Conv2d(128, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.conv2_3 = nn.Sequential(
            nn.Conv2d(64, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.conv3_3 = nn.Sequential(
            nn.Conv2d(32, Classes, 1),
        )

    def forward(self, fused_ss1, fused_ss2, fused_ss3):
        fused_ss1 = self.conv1_1(fused_ss1)
        fused_ss1 = self.conv2_1(fused_ss1)
        rep_1 = self.conv3_1(fused_ss1)
        cls_1 = rep_1.view(rep_1.size(0), -1)
        cls_1 = F.softmax(cls_1, dim=1)

        fused_ss2 = self.conv1_2(fused_ss2)
        fused_ss2 = self.conv2_2(fused_ss2)
        rep_2 = self.conv3_2(fused_ss2)
        cls_2 = rep_2.view(rep_2.size(0), -1)
        cls_2 = F.softmax(cls_2, dim=1)

        fused_ss3 = self.conv1_3(fused_ss3)
        fused_ss3 = self.conv2_3(fused_ss3)
        rep_3 = self.conv3_3(fused_ss3)
        cls_3 = rep_3.view(rep_3.size(0), -1)
        cls_3 = F.softmax(cls_3, dim=1)

        return cls_1, cls_2, cls_3


# Decoupling classification and mapping
class Mapper(nn.Module):
    def __init__(self, Classes):
        super(Mapper, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(128, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x):
        x = self.conv1(x)  # 64*64*4*4
        mapping_x = self.conv2(x)  # 64*32*1*1

        return mapping_x


class Network(nn.Module):
    def __init__(self, l1, l2, Classes):
        super(Network, self).__init__()
        self.encoder = Encoder(l1, l2)  # Spectral-spatial modulation encoder
        self.classifier = Classifier(Classes)  # Multi-scale classification head
        self.mapper = Mapper(Classes)  # Deep mapping head

    def forward(self, h_spa, l_spa, h_spe):
        # Multi-modal spectral-spatial fused encoding feature
        fused_ss1, fused_ss2, fused_ss3 = self.encoder(h_spa, l_spa, h_spe)
        # Multi-scale differentiated classification
        cls_1, cls_2, cls_3 = self.classifier(fused_ss1, fused_ss2, fused_ss3)
        # Deep feature mapping
        mapping_3 = self.mapper(fused_ss3)

        return cls_1, cls_2, cls_3, mapping_3


# Reliable pseudo-label contrastive learning
class RPCLoss(nn.Module):
    """
        Input: mapping features [b1, dim, h, w], reliable pseudo-labels [b1]
        Output: loss value
        b1: reliable sample size
    """

    def __init__(self, margin=2.0):
        super(RPCLoss, self).__init__()
        self.margin = margin

    def similarity_metric (self, x, y):
        """Calculating Euclidean distance between tensors"""
        m, n = x.size(0), y.size(0)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        dist.addmm_(x, y.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()
        return dist

    def forward(self, features, labels):
        device = torch.device('cuda')
        if features.shape[0] == 0:  # the case of null
            loss_r = torch.zeros(1).to(device)
        else:
            features = features.view(features.shape[0], -1)  # [b1, dim, h, w]->[b1, dim*h*w]
            b1 = features.shape[0]  # [b1]
            similarity = self.similarity_metric(features, features)  # Similarity matrix [b1, b1]
            labels = labels.contiguous().view(-1, 1)  # # [b1, 1]
            mask_p = torch.eq(labels, labels.T).float().to(device)  # Positive sample mask
            mask_n = torch.ones(b1, b1).to(device)
            mask_n = torch.sub(mask_n, mask_p)  # Negative sample mask
            loss_p = torch.pow(torch.mul(similarity, mask_p), 2)  # Positive sample loss, minimize intra-class distance
            loss_n = torch.pow(torch.clamp(self.margin - similarity, min=0.0), 2)  # Maximize inter-class distance
            loss_n = torch.mul(loss_n, mask_n)  # Negative sample loss
            loss_r = torch.add(loss_p, loss_n)
            loss_r = torch.mean(loss_r)  # Reliable pseudo-label contrastive learning loss

        return loss_r


# Unreliable pseudo-label contrastive learning
class UPCLoss(nn.Module):
    """
        Input: mapping features [b2, dim, h, w], unreliable pseudo-label ranking [b2, num_class]
               num_class [num_class], positive sample threshold choice [b2]
        Output: loss value
        b2: unreliable sample size
        num_class: number of categories
    """

    def __init__(self):
        super(UPCLoss, self).__init__()

    def similarity_metric (self, x, y):
        """Calculating Euclidean distance between tensors"""
        m, n = x.size(0), y.size(0)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        dist.addmm_(x, y.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()
        return dist

    def forward(self, features, ranking, num_class, choice):
        device = torch.device('cuda')
        if features.shape[0] == 0:  # the case of null
            loss_u = torch.zeros(1).to(device)
        else:
            features = features.view(features.shape[0], -1)  # [b2, dim, h, w]->[b2, dim*h*w]
            b2 = features.shape[0]  # [b2]
            similarity = self.similarity_metric(features, features)  # Similarity matrix [b2, b2]
            labels_u = ranking[:, 0].contiguous().view(-1, 1)  # unreliable pseudo-labels [b2, 1]
            mask = torch.eq(labels_u, labels_u.T).float().to(device)  # Initial mask
            for i in range(b2):
                dic_p = torch.arange(1, 0, -1 / (choice[i]))  # Positive sample weight dictionary
                dic_n = torch.arange(-1 / (num_class - choice[i]), -1 - 1 / (num_class - choice[i]),
                                      -1 / (num_class - choice[i]))  # Negative sample weight dictionary
                dic_s = torch.cat((dic_p, dic_n)).to(device)  # Overall soft weight dictionary
                ranking_i = ranking[i]  # i-th sample pseudo-label ranking [num_class]
                for j in range(num_class):
                    index = torch.where(labels_u.squeeze() == ranking_i[j])[0]  # Query
                    mask[i][index] = dic_s[j]  # Value
            mask_soft = mask  # Soft mask
            # 计算损失
            loss_u = torch.pow(torch.mul(similarity, mask_soft), 2)
            loss_u = torch.mean(loss_u)  # Unreliable pseudo-label contrastive learning loss
        return loss_u


def train_network(train_loader, TestPatch1, TestPatch2, TestLabel, LR, EPOCH, l1, l2, Classes, num_train, order,
                  patch_size, num_labelled, factor_lambda):
    cnn = Network(l1=l1, l2=l2, Classes=Classes)
    cnn.cuda()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)  # Optimize all cnn parameters
    loss_fun1 = nn.CrossEntropyLoss()  # Cross entropy loss
    loss_fun2 = RPCLoss()  # Reliable pseudo-label contrastive learning loss
    loss_fun3 = UPCLoss()  # Unreliable pseudo-label contrastive learning loss
    val_acc = []
    BestAcc = 0
    """Pre-training"""
    cnn.train()
    for epoch in range(EPOCH):
        for step, (b_x1, b_x2, b_y) in enumerate(train_loader):  # Supervised train_loader
            b_x1, b_x2, b_y = b_x1.cuda(), b_x2.cuda(), b_y.cuda()  # Move data to GPU
            b_x1_spe = b_x1[:, :, patch_size // 2, patch_size // 2].unsqueeze(-1)  # Construct spectral vectors
            cls_1, cls_2, cls_3, _ = cnn(b_x1, b_x2, b_x1_spe)  # Model output according to labelled train set
            ce_loss = loss_fun1(cls_1, b_y.long()) + loss_fun1(cls_2, b_y.long()) \
                      + loss_fun1(cls_3, b_y.long())  # Multi-scale classification loss
            cnn.zero_grad()  # Reset gradient
            ce_loss.backward()  # Backward
            optimizer.step()  # Update parameters of net

            if step % 500 == 0:
                cnn.eval()

                pred_y = np.empty((len(TestLabel)), dtype='float32')
                number = len(TestLabel) // 100
                for i in range(number):
                    temp1_1 = TestPatch1[i * 100:(i + 1) * 100, :, :, :]
                    temp1_2 = TestPatch2[i * 100:(i + 1) * 100, :, :, :]
                    temp1_1 = temp1_1.cuda()
                    temp1_2 = temp1_2.cuda()
                    temp2 = cnn(temp1_1, temp1_2, temp1_1[:, :, patch_size // 2, patch_size // 2].unsqueeze(-1))[2]
                    temp3 = torch.max(temp2, 1)[1].squeeze()
                    pred_y[i * 100:(i + 1) * 100] = temp3.cpu()
                    del temp1_1, temp1_2, temp2, temp3

                if (i + 1) * 100 < len(TestLabel):
                    temp1_1 = TestPatch1[(i + 1) * 100:len(TestLabel), :, :, :]
                    temp1_2 = TestPatch2[(i + 1) * 100:len(TestLabel), :, :, :]
                    temp1_1 = temp1_1.cuda()
                    temp1_2 = temp1_2.cuda()
                    temp2 = cnn(temp1_1, temp1_2, temp1_1[:, :, patch_size // 2, patch_size // 2].unsqueeze(-1))[2]
                    temp3 = torch.max(temp2, 1)[1].squeeze()
                    pred_y[(i + 1) * 100:len(TestLabel)] = temp3.cpu()
                    del temp1_1, temp1_2, temp2, temp3

                pred_y = torch.from_numpy(pred_y).long()
                accuracy = torch.sum(pred_y == TestLabel).type(torch.FloatTensor) / TestLabel.size(0)
                print('Epoch: ', epoch, '| classify loss: %.6f' % ce_loss.data.cpu().numpy(),
                      '| test accuracy: %.6f' % accuracy)
                val_acc.append(accuracy.data.cpu().numpy())
                # Save the parameters in network
                if accuracy > BestAcc:
                    torch.save(cnn.state_dict(),
                               './log/Muufl_UACL_baseline_pretrain_%d_%d.pkl' % (
                               num_train, order))
                    BestAcc = accuracy
                    best_y = pred_y

                cnn.train()  # Open Batch Normalization and Dropout
    """Balanced sampling"""
    unlabelled_train = {}
    sample_num_perclass = (num_labelled * 5)// Classes
    for cls in range(Classes):
        indices = [j for j, x in enumerate(best_y.ravel().tolist()) if x == cls]
        np.random.shuffle(indices)  # shuffle
        unlabelled_train[cls] = indices[:sample_num_perclass]
    unlabelled_train_fix_indices = []
    for cls in range(Classes):
        unlabelled_train_fix_indices += unlabelled_train[cls]
    UnlabeledPatch1 = TestPatch1[unlabelled_train_fix_indices]
    UnlabeledPatch2 = TestPatch2[unlabelled_train_fix_indices]
    unlabeled_dataset = Data.TensorDataset(UnlabeledPatch1, UnlabeledPatch2)
    unlabeled_loader = Data.DataLoader(unlabeled_dataset, batch_size=64, shuffle=True, num_workers=8, drop_last=True)
    print('Data1 unlabeled samples size and Data2 unlabeled samples size are:', UnlabeledPatch1.shape, 'and',
          UnlabeledPatch2.shape)
    val_acc2 = []
    BestAcc2 = 0
    """Re-training"""
    cnn.train()
    for epoch in range(EPOCH):
        for step, ((b_x1, b_x2, b_y), (b_u1, b_u2)) in enumerate(
                zip(cycle(train_loader), unlabeled_loader)):  # Semi-supervised train_loader
            b_x1, b_x2, b_y = b_x1.cuda(), b_x2.cuda(), b_y.cuda()  # Move data to GPU
            b_u1, b_u2 = b_u1.cuda(), b_u2.cuda()
            b_x1_spe = b_x1[:, :, patch_size // 2, patch_size // 2].unsqueeze(-1)  # Construct spectral vectors
            b_u1_spe = b_u1[:, :, patch_size // 2, patch_size // 2].unsqueeze(-1)
            cls_1, cls_2, cls_3, mapping_3 = cnn(b_x1, b_x2, b_x1_spe)  # Model output according to labelled train set
            loss_cls = loss_fun1(cls_1, b_y.long()) + loss_fun1(cls_2, b_y.long()) \
                      + loss_fun1(cls_3, b_y.long())  # Multi-scale classification loss
            """Uncertainty-aware contrastive learning"""
            cls_u1, cls_u2, cls_u3, mapping_u3 = cnn(b_u1, b_u2, b_u1_spe)  # Model output according to unlabelled train set
            y_1 = torch.max(cls_u1, 1)[1].squeeze()  # 1-th scale pseudo-label
            y_2 = torch.max(cls_u2, 1)[1].squeeze()  # 2-th scale pseudo-label
            y_3 = torch.max(cls_u3, 1)[1].squeeze()  # 3-th scale pseudo-label
            p3_top = cls_u3.topk(k=1, dim=1)[0].squeeze()  # 3-th scale maximum probability
            std = torch.std(cls_u3, dim=1, keepdim=True)  # 3-th scale probability standard deviation
            mean = torch.mean(cls_u3, dim=1, keepdim=True)  # 3-th scale probability mean
            index_reliable = np.array([])
            index_unreliable = np.array([])
            choice_unreliable = np.array([])
            for j in range(cls_u3.shape[0]):
                if (((p3_top[j] - mean[j]) > 3 * std[j]) or (y_1[j] == y_2[j])):  # Uncertainty analysis
                    index_temp = torch.ones((1)) * j
                    index_reliable = np.append(index_reliable, index_temp)  # Easy sample indexes
                else:
                    index_temp = torch.ones((1)) * j
                    index_unreliable = np.append(index_unreliable, index_temp)  # Hard sample indexes
            index_reliable = index_reliable.astype(int)
            index_unreliable = index_unreliable.astype(int)
            mapping_r = mapping_u3[index_reliable]  # Easy sample mapping
            y_r = y_3[index_reliable]
            l_r = loss_fun2(mapping_r, y_r.long())  # Reliable pseudo-label contrastive loss
            label_rank = cls_u3.topk(k=Classes, dim=1)[1].squeeze()
            prob_rank = cls_u3.topk(k=Classes, dim=1)[0].squeeze()
            unreliable_rep = mapping_u3[index_unreliable]  # Hard sample mapping
            label_rank_unreliable = label_rank[index_unreliable]  # Hard sample pseudo-label ranking
            prob_rank_unreliable = prob_rank[index_unreliable]  # Hard sample probability ranking
            mean_unreliable = mean[index_unreliable]  # Hard sample probability mean
            for i in range(prob_rank_unreliable.shape[0]):
                for j in range(prob_rank_unreliable.shape[1]):
                    if prob_rank_unreliable[i][j] < mean_unreliable[i]:  # Threshold anlysis
                        choice_unreliable = np.append(choice_unreliable, j)  # Positive sample threshold choice
                        break
            l_u = loss_fun3(unreliable_rep, label_rank_unreliable.long(), num_class=Classes,
                                               choice=choice_unreliable)  # Unreliable pseudo-label contrastive loss
            loss_con = l_r + l_u  # Contrastive loss
            loss_overall = loss_cls + factor_lambda * loss_con  # Overall loss
            cnn.zero_grad()  # Reset gradient
            loss_overall.backward()  # Backward
            optimizer.step()  # Update parameters of net

            if step % 500 == 0:
                cnn.eval()

                pred_y = np.empty((len(TestLabel)), dtype='float32')
                number = len(TestLabel) // 100
                for i in range(number):
                    temp1_1 = TestPatch1[i * 100:(i + 1) * 100, :, :, :]
                    temp1_2 = TestPatch2[i * 100:(i + 1) * 100, :, :, :]
                    temp1_1 = temp1_1.cuda()
                    temp1_2 = temp1_2.cuda()
                    temp2 = cnn(temp1_1, temp1_2, temp1_1[:, :, patch_size // 2, patch_size // 2].unsqueeze(-1))[2]
                    temp3 = torch.max(temp2, 1)[1].squeeze()
                    pred_y[i * 100:(i + 1) * 100] = temp3.cpu()
                    del temp1_1, temp1_2, temp2, temp3

                if (i + 1) * 100 < len(TestLabel):
                    temp1_1 = TestPatch1[(i + 1) * 100:len(TestLabel), :, :, :]
                    temp1_2 = TestPatch2[(i + 1) * 100:len(TestLabel), :, :, :]
                    temp1_1 = temp1_1.cuda()
                    temp1_2 = temp1_2.cuda()
                    temp2 = cnn(temp1_1, temp1_2, temp1_1[:, :, patch_size // 2, patch_size // 2].unsqueeze(-1))[2]
                    temp3 = torch.max(temp2, 1)[1].squeeze()
                    pred_y[(i + 1) * 100:len(TestLabel)] = temp3.cpu()
                    del temp1_1, temp1_2, temp2, temp3

                pred_y = torch.from_numpy(pred_y).long()
                accuracy = torch.sum(pred_y == TestLabel).type(torch.FloatTensor) / TestLabel.size(0)
                print('Epoch: ', epoch, '| classify loss: %.6f' % ce_loss.data.cpu().numpy(),
                      '| pseudo loss: %.6f' % loss_con.data.cpu().numpy(),
                      '| test accuracy: %.6f' % accuracy)
                val_acc2.append(accuracy.data.cpu().numpy())
                # save the parameters in network
                if accuracy > BestAcc2:
                    torch.save(cnn.state_dict(),
                               './log/Muufl_UACL_baseline_finetune_%d_%d.pkl' % (
                               num_train, order))
                    BestAcc2 = accuracy

                cnn.train()  # Open Batch Normalization and Dropout

    cnn.load_state_dict(torch.load(
        './log/Muufl_UACL_baseline_finetune_%d_%d.pkl' % (
        num_train, order)))
    cnn.eval()

    pred_y = np.empty((len(TestLabel)), dtype='float32')
    number = len(TestLabel) // 100
    for i in range(number):
        temp1_1 = TestPatch1[i * 100:(i + 1) * 100, :, :, :]
        temp1_2 = TestPatch2[i * 100:(i + 1) * 100, :, :, :]
        temp1_1 = temp1_1.cuda()
        temp1_2 = temp1_2.cuda()
        temp2 = cnn(temp1_1, temp1_2, temp1_1[:, :, patch_size // 2, patch_size // 2].unsqueeze(-1))[2]
        temp3 = torch.max(temp2, 1)[1].squeeze()

        pred_y[i * 100:(i + 1) * 100] = temp3.cpu()
        del temp1_1, temp1_2, temp2, temp3

    if (i + 1) * 100 < len(TestLabel):
        temp1_1 = TestPatch1[(i + 1) * 100:len(TestLabel), :, :, :]
        temp1_2 = TestPatch2[(i + 1) * 100:len(TestLabel), :, :, :]
        temp1_1 = temp1_1.cuda()
        temp1_2 = temp1_2.cuda()
        temp2 = cnn(temp1_1, temp1_2, temp1_1[:, :, patch_size // 2, patch_size // 2].unsqueeze(-1))[2]
        temp3 = torch.max(temp2, 1)[1].squeeze()
        pred_y[(i + 1) * 100:len(TestLabel)] = temp3.cpu()
        del temp1_1, temp1_2, temp2, temp3

    pred_y = torch.from_numpy(pred_y).long()
    OA = torch.sum(pred_y == TestLabel).type(torch.FloatTensor) / TestLabel.size(0)

    Classes = np.unique(TestLabel)
    EachAcc = np.empty(len(Classes))

    for i in range(len(Classes)):
        cla = Classes[i]
        right = 0
        sum = 0

        for j in range(len(TestLabel)):
            if TestLabel[j] == cla:
                sum = sum + 1
                # sum += 1

            if TestLabel[j] == cla and pred_y[j] == cla:
                right = right + 1
                # right += 1

        EachAcc[i] = right.__float__() / sum.__float__()
        AA = np.mean(EachAcc)

    print(OA)
    print(EachAcc)
    print(AA)
    return pred_y, val_acc2
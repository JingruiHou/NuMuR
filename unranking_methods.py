import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class RankingDistilLoss(nn.Module):
    def __init__(self,
                 cfg,
                 _loss_retain_pos_fn=None,
                 _loss_retain_neg_fn=None,
                 _loss_forget_neg_fn=None,
                 _loss_forget_pos_fn=None,
                 quantile_neg=0.0):
        super(RankingDistilLoss, self).__init__()
        # 自定义的损失函数
        self._loss_retain_pos_fn = _loss_retain_pos_fn if _loss_retain_pos_fn else self.hinge_loss
        self._loss_retain_neg_fn = _loss_retain_neg_fn if _loss_retain_neg_fn else self.hinge_loss
        self._loss_forget_neg_fn = _loss_forget_neg_fn if _loss_forget_neg_fn else self.hinge_loss
        self._loss_forget_pos_fn = _loss_forget_pos_fn if _loss_forget_pos_fn else self.hinge_loss
        # 分位数参数
        self.quantile_neg = quantile_neg
        self.config = cfg

    def forward(self, student_pos_out, student_neg_outs, student_subs_out, teacher_pos_out, teacher_neg_outs, labels):
        batch_size = labels.size(0)
        # Calculate the mean of teacher negative outputs
        teacher_neg_mean = torch.quantile(torch.stack(teacher_neg_outs), self.quantile_neg, interpolation='linear', dim=0)
        # Create boolean masks for negative and positive samples
        mask_neg = (labels == 0)
        mask_pos = (labels == 1)
        bias = torch.min(mask_pos.sum()//mask_neg.sum(), torch.tensor(batch_size, device=labels.device))
        if self.config.get('loss_bias') and self.config['loss_bias'] > 0:
            bias = self.config['loss_bias'] * bias
        else:
            bias = 1
        # Initialize loss
        distil_loss = torch.tensor(1e-8, device=labels.device)
        # Calculate MSE loss for positive samples
        if mask_pos.any():
            for student_neg_out, teacher_neg_out in zip(student_neg_outs, teacher_neg_outs):
                # 1, 0.5
                distil_loss += self._loss_retain_neg_fn(student_neg_out[mask_pos], teacher_neg_out[mask_pos],
                                                        b=self.config['hinge_loss_retain_neg_b'])
            # -1 ,-2
            distil_loss += self._loss_retain_pos_fn(student_pos_out[mask_pos], teacher_pos_out[mask_pos],
                                                a=self.config['hinge_loss_retain_pos_a'], b=self.config['hinge_loss_retain_pos_b'])

        # Calculate MSE loss for negative samples
        if mask_neg.any():
            # 1, 0.1
            distil_loss += bias * self._loss_forget_pos_fn(student_pos_out[mask_neg], teacher_neg_mean[mask_neg],
                                                    b=self.config['hinge_loss_forget_pos_b'])
            distil_loss += bias * self._loss_forget_neg_fn(student_subs_out[mask_neg], teacher_pos_out[mask_neg],
                                                    a=self.config['hinge_loss_forget_neg_a'], b=self.config['hinge_loss_forget_neg_b'])
        distil_loss /= batch_size
        return distil_loss

    @staticmethod
    def mse_loss(x, y):
        return F.mse_loss(x, y, reduction='sum')

    @staticmethod
    def hinge_loss(x, y, a=1, b=1, margin=0.0):
        return torch.sum(torch.clamp(a*x - b*y + margin, min=0))

    @staticmethod
    def exponential_loss(x, y):
        return torch.sum(torch.exp(x - y))


def bad_ranking_teacher_loss(student_pos_out, teacher_pos_out, teacher_neg_outs, labels):
    batch_size = labels.size(0)
    # Calculate the mean of teacher negative outputs
    teacher_neg_mean = torch.mean(torch.stack(teacher_neg_outs), dim=0)
    # Use boolean masks to select positive and negative samples
    mask_neg = (labels == 0)
    mask_pos = (labels == 1)
    # Calculate the losses
    mse_loss = torch.tensor(0.0, device=labels.device)
    # Loss for negative labels
    if mask_neg.sum() > 0:
        mse_loss_neg = F.mse_loss(student_pos_out[mask_neg], teacher_neg_mean[mask_neg],
                                  reduction='sum')
        mse_loss += mse_loss_neg
    # Loss for positive labels
    if mask_pos.sum() > 0:
        mse_loss_pos = F.mse_loss(student_pos_out[mask_pos], teacher_pos_out[mask_pos], reduction='sum')
        mse_loss += mse_loss_pos
    # Normalize by batch size
    mse_loss /= batch_size
    return mse_loss


def ranking_teacher_loss(student_pos_out, student_neg_outs, teacher_pos_out, teacher_neg_outs, labels, quantile_neg=0.5):
    batch_size = labels.size(0)
    # Calculate the mean of teacher negative outputs
    teacher_neg_mean = torch.quantile(torch.stack(teacher_neg_outs), quantile_neg, interpolation='linear', dim=0)
    # Create boolean masks for negative and positive samples
    mask_neg = (labels == 0)
    mask_pos = (labels == 1)
    # Initialize loss
    mse_loss = torch.tensor(0.0, device=labels.device)
    # Calculate MSE loss for positive samples
    if mask_pos.any():
        mse_loss_retain = F.mse_loss(student_pos_out[mask_pos], teacher_pos_out[mask_pos], reduction='sum')
        mse_loss += mse_loss_retain
        for student_neg_out, teacher_neg_out in zip(student_neg_outs, teacher_neg_outs):
            mse_loss_retain_neg = F.mse_loss(student_neg_out[mask_pos], teacher_neg_out[mask_pos], reduction='sum')
            mse_loss += mse_loss_retain_neg / len(student_neg_outs)
    # Calculate MSE loss for negative samples
    if mask_neg.any():
        mse_loss_forget_pos = F.mse_loss(student_pos_out[mask_neg], teacher_neg_mean[mask_neg],
                          reduction='sum')
        mse_loss += mse_loss_forget_pos
        for student_neg_out in student_neg_outs:
            mse_loss_forget_neg = F.mse_loss(student_neg_out[mask_neg], teacher_pos_out[mask_neg], reduction='sum')
            mse_loss += mse_loss_forget_neg / len(student_neg_outs)
    # Normalize loss by batch size
    mse_loss /= batch_size
    return mse_loss


class CoCoLDistillationLoss(nn.Module):
    def __init__(self, margin=0.0, ratio=1):
        if not margin:
            margin = 0.0
        if not ratio:
            ratio = 1
        super(CoCoLDistillationLoss, self).__init__()
        self.margin = margin
        self.ratio = ratio

    def forward(self, score_t, score_s, loss_type='contrastive'):
        if loss_type == 'contrastive':
            score_diff = (score_s - self.ratio * score_t + self.margin) / (score_t + score_s)
            loss = torch.mean(torch.relu(score_diff))
        elif loss_type == 'consistent':
            distance_retaining_p = torch.abs((score_t - score_s) / (score_t + score_s))
            loss = torch.mean(distance_retaining_p)
        return loss



class CoCoLDistillationLossV2(nn.Module):
    def __init__(self, margin=0.0, ratio=1):
        if not margin:
            margin = 0.0
        if not ratio:
            ratio = 1
        super(CoCoLDistillationLoss, self).__init__()
        self.margin = margin
        self.ratio = ratio

    def forward(self, score_t, score_s, loss_type='diff_1'):
        if loss_type == 'diff_1':
            score_diff = (score_s - self.ratio * score_t + self.margin) / (score_t + score_s)
            loss = torch.mean(torch.relu(score_diff))
        elif loss_type == 'sim_1':
            distance_retaining_p = torch.abs((score_t - score_s) / (score_t + score_s))
            loss = torch.mean(distance_retaining_p)
        return loss

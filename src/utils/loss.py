import torch
import torch.nn as nn
import torch.nn.functional as F

bce_loss = nn.BCELoss()
mse_loss = nn.MSELoss(reduction = "mean")

def bce_loss_calc(gt_feature_maps, gt):
    sum_loss = 0
    for idx, output in enumerate(gt_feature_maps):
        component = F.upsample(output, gt.shape[2:], mode='bilinear')
        loss = bce_loss(component, gt)
        sum_loss += loss
    return sum_loss

def feature_sync(gt_outputs, u2net_outputs):
    loss_lst = []
    for idx, gt_output in enumerate(gt_outputs):
        loss = mse_loss(gt_output, u2net_outputs[idx])
        loss_lst.append(loss)

    loss = sum(loss_lst)
    return loss

    loss = nn.L1Loss()
    return loss(input_data, target_data)

def mae_loss(input_data, target_data):
    loss = nn.L1Loss()
    return loss(input_data, target_data)
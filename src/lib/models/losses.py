# ------------------------------------------------------------------------------
# Portions of this code are from
# CornerNet (https://github.com/princeton-vl/CornerNet)
# Copyright (c) 2018, University of Michigan
# Licensed under the BSD 3-Clause License
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from .utils import _transpose_and_gather_feat
import torch.nn.functional as F
from PIL import Image, ImageDraw
import numpy as np
import math

def _slow_neg_loss(pred, gt):
  '''focal loss from CornerNet'''
  pos_inds = gt.eq(1)
  neg_inds = gt.lt(1)

  neg_weights = torch.pow(1 - gt[neg_inds], 4)

  loss = 0
  pos_pred = pred[pos_inds]
  neg_pred = pred[neg_inds]

  pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
  neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights

  num_pos  = pos_inds.float().sum()
  pos_loss = pos_loss.sum()
  neg_loss = neg_loss.sum()

  if pos_pred.nelement() == 0:
    loss = loss - neg_loss
  else:
    loss = loss - (pos_loss + neg_loss) / num_pos
  return loss


def _neg_loss(pred, gt):
  ''' Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
  '''
  pos_inds = gt.eq(1).float()
  neg_inds = gt.lt(1).float()

  neg_weights = torch.pow(1 - gt, 4)

  loss = 0

  pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
  neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

  num_pos  = pos_inds.float().sum()
  pos_loss = pos_loss.sum()
  neg_loss = neg_loss.sum()

  if num_pos == 0:
    loss = loss - neg_loss
  else:
    loss = loss - (pos_loss + neg_loss) / num_pos
  return loss

def _not_faster_neg_loss(pred, gt):
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()    
    num_pos  = pos_inds.float().sum()
    neg_weights = torch.pow(1 - gt, 4)

    loss = 0
    trans_pred = pred * neg_inds + (1 - pred) * pos_inds
    weight = neg_weights * neg_inds + pos_inds
    all_loss = torch.log(1 - trans_pred) * torch.pow(trans_pred, 2) * weight
    all_loss = all_loss.sum()

    if num_pos > 0:
        all_loss /= num_pos
    loss -=  all_loss
    return loss

def _slow_reg_loss(regr, gt_regr, mask):
    num  = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr)

    regr    = regr[mask]
    gt_regr = gt_regr[mask]
    
    regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr,  reduction='sum')
    regr_loss = regr_loss / (num + 1e-4)
    return regr_loss

def _reg_loss(regr, gt_regr, mask):
  ''' L1 regression loss
    Arguments:
      regr (batch x max_objects x dim)
      gt_regr (batch x max_objects x dim)
      mask (batch x max_objects)
  '''
  num = mask.float().sum()
  mask = mask.unsqueeze(2).expand_as(gt_regr).float()

  regr = regr * mask
  gt_regr = gt_regr * mask
    
  regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr,  reduction='sum')
  regr_loss = regr_loss / (num + 1e-4)
  return regr_loss

class FocalLoss(nn.Module):
  '''nn.Module warpper for focal loss'''
  def __init__(self):
    super(FocalLoss, self).__init__()
    self.neg_loss = _neg_loss

  def forward(self, out, target):
    return self.neg_loss(out, target)

class RegLoss(nn.Module):
  '''Regression loss for an output tensor
    Arguments:
      output (batch x dim x h x w)
      mask (batch x max_objects)
      ind (batch x max_objects)
      target (batch x max_objects x dim)
  '''
  def __init__(self):
    super(RegLoss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _transpose_and_gather_feat(output, ind)
    loss = _reg_loss(pred, target, mask)
    return loss

class RegL1Loss(nn.Module):
  def __init__(self):
    super(RegL1Loss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _transpose_and_gather_feat(output, ind)
    mask = mask.unsqueeze(2).expand_as(pred).float()
    # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
    loss = F.l1_loss(pred * mask, target * mask,  reduction='sum')
    loss = loss / (mask.sum() + 1e-4)
    return loss


class RegL1PolyLoss(nn.Module):
    def __init__(self):
        super(RegL1PolyLoss, self).__init__()

    def forward(self, output, mask, ind, target, freq_mask, hm = None):
        pred = _transpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        # mask = mask.expand_as(pred).float()
        # norm_pred = pred - torch.min(pred)
        # norm_pred /= (torch.max(pred) + 1e-4)
        # norm_target = pred - torch.min(target)
        # norm_target /= (torch.max(target) + 1e-4)
        loss = F.l1_loss(pred * mask, target * mask, reduction='sum')
        # loss = nn.MSELoss(reduction='sum')(pred * mask, target * mask)
        # loss = nn.SmoothL1Loss(reduction='sum')(pred * mask, target * mask)
        # loss *= torch.max(target)
        # loss = F.l1_loss(pred * mask, target * mask, reduction='sum')
        # loss *= (size_norm.sum())
        loss = loss / (mask.sum() + 1e-4)
        # loss = loss / (freq_mask.mean() + 1e-4)
        # print(freq_mask)
        return loss

class RegL1PolyPolarLoss(nn.Module):
    def __init__(self):
        super(RegL1PolyPolarLoss, self).__init__()

    def forward(self, output, mask, ind, target, freq_mask, hm = None):

        WEIGHT_ANGLE = 10
        mask_angles = torch.FloatTensor([1,0]*output.shape[1])
        
        pred = _transpose_and_gather_feat(output, ind)
        mask_angles = mask_angles.unsqueeze(0).unsqueeze(2).expand_as(pred)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        loss = F.l1_loss(pred * mask*mask_angles, target * mask*mask_angles, reduction='sum')
        loss += F.l1_loss(pred * mask*WEIGHT_ANGLE*(1-mask_angles), target * mask*WEIGHT_ANGLE*(1-mask_angles), reduction='sum')
        loss = loss / (mask.sum() + 1e-4)
        return loss


class AreaPolyLoss(nn.Module):
    def __init__(self):
        super(AreaPolyLoss, self).__init__()

    def forward(self, output, mask, ind, target, centers):
        pred = _transpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        loss = 0

        for batch in range(output.shape[0]):
            polygon_mask = Image.new('L', (output.shape[-1], output.shape[-2]), 0)
            poly_points = []
            for i in range(0, pred[batch].shape[0]):  # nbr objects
                for j in range(0, pred[batch].shape[1] - 1, 2):  # points
                    poly_points.append((int(pred[batch][i][j] + centers[batch][i][0]),
                                        int(pred[batch][i][j+1] + centers[batch][i][1])))

            ImageDraw.Draw(polygon_mask).polygon(poly_points, outline=0, fill=255)
            polygon_mask = torch.Tensor(np.array(polygon_mask)).cuda()
            loss += nn.MSELoss()(polygon_mask, target[batch])
        # loss = F.l1_loss(pred * mask, target * mask, reduction='sum')
        loss = loss / (mask.sum() + 1e-4)
        return loss

class IoUPolyLoss(nn.Module):
    def __init__(self):
        super(IoUPolyLoss, self).__init__()

    def forward(self, output, mask, ind, target, freq_mask = None, hm = None):
        """
        Parameters:
            output: output of polygon head
              [batch_size, 2*nb_vertices, nb_max_objects, nb_heads]
            mask: selected objects
              [batch_size, nb_max_objects]
            ind:
              [batch_size, nb_max_objects]
            target: ground-truth for the polygons
              [batch_size, 2*nb_vertices, nb_max_objects]
        Returns:
            loss: scalar
        """
        #print(output.shape) #[batch_size, 32, nb_objects (=128), head_conv]
        #print(mask) #[batch_size, nb_objects]
        #print(ind.shape) #[batch_size, nb_objects]
        #print(target.shape) #[batch_size, nb_objects, 32]
        #print(freq_mask.shape) #[batch_size]

        pred = _transpose_and_gather_feat(output, ind) #[batch_size, nb_objects, 32]
        #mask = mask.unsqueeze(2).expand_as(pred).float() #[batch_size, nb_objects, 32]
        #print(pred.shape)
        #print(mask.shape)
        loss = 0

        OFFSET = 100

        predictions = False

        #print(mask.sum())

        for batch in range(output.shape[0]):

            for i in range(0, pred[batch].shape[0]):  # nbr objects

                if mask[batch][i]:
                    #print("object")
                    polygon_mask_pred = Image.new('L', (output.shape[-1], output.shape[-2]), 0)
                    poly_points_pred = []

                    polygon_mask_gt = Image.new('L', (output.shape[-1], output.shape[-2]), 0)
                    poly_points_gt = []

                    predictions = True

                    for j in range(0, pred[batch].shape[1] - 1, 2):  # points
                        #print(j)
                        poly_points_pred.append((int(pred[batch][i][j])+OFFSET,
                                            int(pred[batch][i][j+1])+OFFSET))
                        poly_points_gt.append((int(target[batch][i][j])+OFFSET,
                                            int(target[batch][i][j+1])+OFFSET))

                    #print(len(poly_points_pred))
                    #print(poly_points_pred)
                    #print(poly_points_gt)
                    ImageDraw.Draw(polygon_mask_pred).polygon(poly_points_pred, outline=0, fill=255)
                    #polygon_mask_pred.show()
                    polygon_mask_pred = torch.Tensor(np.array(polygon_mask_pred)).cuda()

                    ImageDraw.Draw(polygon_mask_gt).polygon(poly_points_gt, outline=0, fill=255)
                    #polygon_mask_gt.show()
                    polygon_mask_gt = torch.Tensor(np.array(polygon_mask_gt)).cuda()

                    #loss += nn.MSELoss()(polygon_mask, target[batch])
                    intersection = torch.sum((polygon_mask_pred + polygon_mask_gt) == 510)
                    #print(intersection)
                    union = torch.sum(polygon_mask_pred != 0) + torch.sum(polygon_mask_gt != 0) - intersection
                    #print("aire pred ", torch.sum(polygon_mask_pred != 0))
                    #print("aire gt ", torch.sum(polygon_mask_gt != 0))
                    #print(union)
                    loss += intersection/(union+ 1e-4)
                    #print(loss.shape)
                    #print(loss)

            if not predictions: #no centers predicted
                loss = 0.0

        #loss = loss.float()
        #print(loss)

        #loss = F.l1_loss(pred * mask, target * mask, reduction='sum')

        #loss = loss / (mask.sum() + 1e-4)
        loss = 1 - loss / (mask.sum() + 1e-4)
        #print("final loss ", loss)
        return loss


class IoUPolyPolarLoss(nn.Module):
    def __init__(self):
        super(IoUPolyLoss, self).__init__()

    def forward(self, output, mask, ind, target, freq_mask = None, hm = None):
        """
        Parameters:
            output: output of polygon head
              [batch_size, 2*nb_vertices, nb_max_objects, nb_heads]
            mask: selected objects
              [batch_size, nb_max_objects]
            ind:
              [batch_size, nb_max_objects]
            target: ground-truth for the polygons
              [batch_size, 2*nb_vertices, nb_max_objects]
            hm: output of heatmap head
              [batch_size, nb_categories, height, width]
        Returns:
            loss: scalar
        """
        #print(output.shape) #[batch_size, 32, nb_objects (=128), head_conv]
        #print(mask) #[batch_size, nb_objects]
        #print(ind.shape) #[batch_size, nb_objects]
        #print(target.shape) #[batch_size, nb_objects, 32]
        #print(freq_mask.shape) #[batch_size]

        pred = _transpose_and_gather_feat(output, ind) #[batch_size, nb_objects, 32]
        #mask = mask.unsqueeze(2).expand_as(pred).float() #[batch_size, nb_objects, 32]
        #print(pred.shape)
        #print(mask.shape)
        loss = 0

        OFFSET = 100

        predictions = False

        #print(mask.sum())

        for batch in range(output.shape[0]):

            for i in range(0, pred[batch].shape[0]):  # nbr objects

                if mask[batch][i]:
                    #print("object")
                    polygon_mask_pred = Image.new('L', (output.shape[-1], output.shape[-2]), 0)
                    poly_points_pred = []

                    polygon_mask_gt = Image.new('L', (output.shape[-1], output.shape[-2]), 0)
                    poly_points_gt = []

                    predictions = True

                    for j in range(0, pred[batch].shape[1] - 1, 2):  # points
                        #print(j)
                        poly_points_pred.append((OFFSET+pred[batch][i][j]*math.cos(pred[batch][i][j+1]),
                                            OFFSET+pred[batch][i][j]*math.sin(pred[batch][i][j+1])))
                        poly_points_gt.append((target[batch][i][j]*math.cos(target[batch][i][j+1])+OFFSET,
                                            target[batch][i][j]*math.sin(target[batch][i][j+1])+OFFSET))

                    #print(len(poly_points_pred))
                    #print(poly_points_pred)
                    #print(poly_points_gt)
                    ImageDraw.Draw(polygon_mask_pred).polygon(poly_points_pred, outline=0, fill=255)
                    #polygon_mask_pred.show()
                    polygon_mask_pred = torch.Tensor(np.array(polygon_mask_pred)).cuda()

                    ImageDraw.Draw(polygon_mask_gt).polygon(poly_points_gt, outline=0, fill=255)
                    #polygon_mask_gt.show()
                    polygon_mask_gt = torch.Tensor(np.array(polygon_mask_gt)).cuda()

                    #loss += nn.MSELoss()(polygon_mask, target[batch])
                    intersection = torch.sum((polygon_mask_pred + polygon_mask_gt) == 510)
                    #print(intersection)
                    union = torch.sum(polygon_mask_pred != 0) + torch.sum(polygon_mask_gt != 0) - intersection
                    #print("aire pred ", torch.sum(polygon_mask_pred != 0))
                    #print("aire gt ", torch.sum(polygon_mask_gt != 0))
                    #print(union)
                    loss += intersection/(union+ 1e-4)
                    #print(loss.shape)
                    #print(loss)

            if not predictions: #no centers predicted
                loss = 0.0

        #loss = loss.float()
        #print(loss)

        #loss = F.l1_loss(pred * mask, target * mask, reduction='sum')

        #loss = loss / (mask.sum() + 1e-4)
        loss = 1 - loss / (mask.sum() + 1e-4)
        #print("final loss ", loss)
        return loss

class NormRegL1Loss(nn.Module):
  def __init__(self):
    super(NormRegL1Loss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _transpose_and_gather_feat(output, ind)
    mask = mask.unsqueeze(2).expand_as(pred).float()
    # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
    pred = pred / (target + 1e-4)
    target = target * 0 + 1
    loss = F.l1_loss(pred * mask, target * mask, reduction='sum')
    loss = loss / (mask.sum() + 1e-4)
    return loss

class RegWeightedL1Loss(nn.Module):
  def __init__(self):
    super(RegWeightedL1Loss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _transpose_and_gather_feat(output, ind)
    mask = mask.float()
    # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
    loss = F.l1_loss(pred * mask, target * mask,  reduction='sum')
    loss = loss / (mask.sum() + 1e-4)
    return loss

class L1Loss(nn.Module):
  def __init__(self):
    super(L1Loss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _transpose_and_gather_feat(output, ind)
    mask = mask.unsqueeze(2).expand_as(pred).float()
    loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
    return loss

class BinRotLoss(nn.Module):
  def __init__(self):
    super(BinRotLoss, self).__init__()
  
  def forward(self, output, mask, ind, rotbin, rotres):
    pred = _transpose_and_gather_feat(output, ind)
    loss = compute_rot_loss(pred, rotbin, rotres, mask)
    return loss

def compute_res_loss(output, target):
    return F.smooth_l1_loss(output, target, reduction='elementwise_mean')

# TODO: weight
def compute_bin_loss(output, target, mask):
    mask = mask.expand_as(output)
    output = output * mask.float()
    return F.cross_entropy(output, target, reduction='elementwise_mean')

def compute_rot_loss(output, target_bin, target_res, mask):
    # output: (B, 128, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos, 
    #                 bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
    # target_bin: (B, 128, 2) [bin1_cls, bin2_cls]
    # target_res: (B, 128, 2) [bin1_res, bin2_res]
    # mask: (B, 128, 1)
    # import pdb; pdb.set_trace()
    output = output.view(-1, 8)
    target_bin = target_bin.view(-1, 2)
    target_res = target_res.view(-1, 2)
    mask = mask.view(-1, 1)
    loss_bin1 = compute_bin_loss(output[:, 0:2], target_bin[:, 0], mask)
    loss_bin2 = compute_bin_loss(output[:, 4:6], target_bin[:, 1], mask)
    loss_res = torch.zeros_like(loss_bin1)
    if target_bin[:, 0].nonzero().shape[0] > 0:
        idx1 = target_bin[:, 0].nonzero()[:, 0]
        valid_output1 = torch.index_select(output, 0, idx1.long())
        valid_target_res1 = torch.index_select(target_res, 0, idx1.long())
        loss_sin1 = compute_res_loss(
          valid_output1[:, 2], torch.sin(valid_target_res1[:, 0]))
        loss_cos1 = compute_res_loss(
          valid_output1[:, 3], torch.cos(valid_target_res1[:, 0]))
        loss_res += loss_sin1 + loss_cos1
    if target_bin[:, 1].nonzero().shape[0] > 0:
        idx2 = target_bin[:, 1].nonzero()[:, 0]
        valid_output2 = torch.index_select(output, 0, idx2.long())
        valid_target_res2 = torch.index_select(target_res, 0, idx2.long())
        loss_sin2 = compute_res_loss(
          valid_output2[:, 6], torch.sin(valid_target_res2[:, 1]))
        loss_cos2 = compute_res_loss(
          valid_output2[:, 7], torch.cos(valid_target_res2[:, 1]))
        loss_res += loss_sin2 + loss_cos2
    return loss_bin1 + loss_bin2 + loss_res

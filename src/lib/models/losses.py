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
import matplotlib.pyplot as plt
import seaborn as sns
import time
from utils.image import draw_ellipse_gaussian

DRAW = False

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

def dice_score(inputs, targets):
    inputs = inputs.sigmoid()
    numerator = 2*torch.matmul(inputs, targets.t())
    denominator = (inputs * inputs).sum(-1)[:, None] + (targets * targets).sum(-1)
    score = numerator / (denominator + 1e-4)
    return score

def create_mask(output, pred, target, batch, num_object, rep):


  SUBPIXEL = 1
  OFFSET_Y = SUBPIXEL*output.shape[-1]//4 ###
  OFFSET_X = SUBPIXEL*output.shape[-2]//4 ###
  i = num_object


  #print("object")
  polygon_mask_pred = Image.new('L', (SUBPIXEL*output.shape[-1], SUBPIXEL*output.shape[-2]), 0) ###
  poly_points_pred = []
  #print(polygon_mask_pred.size)

  polygon_mask_gt = Image.new('L', (SUBPIXEL*output.shape[-1], SUBPIXEL*output.shape[-2]), 0) ###
  poly_points_gt = []


  for j in range(0, pred[batch].shape[1] - 1, 2):  # points
      #print(j)
      if rep =='polar':
        poly_points_pred.append((OFFSET_X+pred[batch][i][j]*math.cos(pred[batch][i][j+1]),
                            OFFSET_Y+pred[batch][i][j]*math.sin(pred[batch][i][j+1])))
        poly_points_gt.append((target[batch][i][j]*math.cos(target[batch][i][j+1])+OFFSET_X,
                            target[batch][i][j]*math.sin(target[batch][i][j+1])+OFFSET_Y))
      elif rep=='cartesian':
        poly_points_pred.append((SUBPIXEL*pred[batch][i][j]+OFFSET_X,
                            SUBPIXEL*pred[batch][i][j+1]+OFFSET_Y)) ###
        poly_points_gt.append((SUBPIXEL*target[batch][i][j]+OFFSET_X,
                            SUBPIXEL*target[batch][i][j+1]+OFFSET_Y)) ###
      elif rep=='polar_fixed':
        fixed_angle = 2*3.14 - 2*3.14/pred[batch].shape[1]*j

        #print(fixed_angle)

        poly_points_pred.append((OFFSET_X+pred[batch][i][j]*math.cos(fixed_angle),
                            OFFSET_Y+pred[batch][i][j]*math.sin(fixed_angle)))
        poly_points_gt.append((target[batch][i][j]*math.cos(target[batch][i][j+1])+OFFSET_X,
                            target[batch][i][j]*math.sin(target[batch][i][j+1])+OFFSET_Y))

  #print(poly_points_pred)
  #print(poly_points_gt)
  ImageDraw.Draw(polygon_mask_pred).polygon(poly_points_pred, outline=0, fill=255)
  #polygon_mask_pred.show()
  polygon_mask_pred = torch.Tensor(np.array(polygon_mask_pred)).cuda()

  ImageDraw.Draw(polygon_mask_gt).polygon(poly_points_gt, outline=0, fill=255)
  #polygon_mask_gt.show()
  polygon_mask_gt = torch.Tensor(np.array(polygon_mask_gt)).cuda()

  #time.sleep(5)

  return polygon_mask_pred, polygon_mask_gt

def differentiable_gaussian(H, W, centers, radius, ceiling = 'sigmoid'):

  device = centers.device

  N = centers.shape[-1]//2

  centers = centers.view(centers.shape[0], centers.shape[1], N, 2)
  # batch_size, nb_max_obj, nb_points, 2
  #print(centers.shape)

  centers = centers.unsqueeze(2).unsqueeze(2)
  # batch_size, nb_max_obj, 1, 1, nb_points, 2
  #print(centers.shape)

  centers = centers.repeat(1, 1, H, W, 1, 1)
  # batch_size, nb_max_obj, H, W, nb_points, 2
  #print(centers.shape)

  indexes = torch.FloatTensor([[[i,j] for i in range(W)] for j in range(H)]).to(device)
  #print(indexes.shape)
  indexes = indexes.unsqueeze(-2)
  # batch_size, nb_max_obj, H, W, nb_points, 2
  indexes = indexes.expand(centers.shape[0], centers.shape[1], H, W, N, 2)
  #print(indexes.shape)

  #print(indexes[0,0,:,:,1,1])
  #print(indexes[0,0,:,:,1,0])

  centers = centers - indexes

  #print(centers.grad_fn)

  centers = - torch.pow(centers,2)

  centers = centers.sum(-1)
  # batch_size, nb_max_obj, H, W, nb_points
  #print(centers.shape)

  radius = radius.unsqueeze(-1).unsqueeze(-1)
  # batch_size, nb_max_obj, H, W, nb_points, 2
  radius = radius.expand(radius.shape[0], radius.shape[1], H, W,N)
  #print(radius.shape)

  centers = torch.exp(centers/(2*torch.pow(radius,2)))

  #print(centers.grad_fn)

  centers = centers.sum(-1)
  # batch_size, nb_max_obj, H, W
  #print(centers.shape)

  if ceiling == 'sigmoid':
    centers = torch.sigmoid(centers)
  elif ceiling == 'clamp':
    centers = torch.clamp(centers, 0, 1)

  return centers


def individual_gaussian(heatmap, centers, radius):

  device = heatmap.device

  H, W = heatmap.shape

  centers = centers.view(centers.shape[0], centers.shape[1], centers.shape[-1]//2, 2)
  # batch_size, nb_max_obj, nb_points, 2
  print(centers.shape)

  centers = centers.unsqueeze(2).unsqueeze(2)
  # batch_size, nb_max_obj, 1, 1, nb_points, 2
  print(centers.shape)

  centers = centers.repeat(1, 1, H, W, 1, 1)


  for k in range(0,centers.shape[0],2):

    #print(centers[k], centers[k+1])

    heatmap += torch.exp( -(torch.pow(torch.FloatTensor([[centers[k]-i for i in range(W)]]).to(device), 2) \
               + torch.pow(torch.FloatTensor([[centers[k+1]-j] for j in range(H)]).to(device), 2))/(2*radius**2))

    """
    for y, line in enumerate(heatmap):
          for x, value in enumerate(line):
              value = ((x - centers[k])**2 + (y-centers[k+1])**2) / ((2*radius)**2)
              #if abs(x-centers[k]) < 10 and abs(y-centers[k+1]) < 10:
              #  print(value)
              #  print(centers[k], centers[k+1])
              #  print(x, y, torch.exp(-value))
              heatmap[y,x] = torch.exp(-value)
              #if torch.exp(-value) >0.1:
              #  print(x, y, torch.exp(-value))
    """

  #ax = sns.heatmap(heatmap.detach().cpu().numpy())
  #plt.show()



  return heatmap

def display_gaussian_image(heatmap, centers, radius):#, peak):

  device = heatmap.device

  #sigma = torch.unsqueeze(torch.unsqueeze(parameters[:,:,-1], 2), 3)
  #print('radius', radius.shape)
  #print('peak', peak.shape)
  sigma = torch.unsqueeze(radius, 2)
  #print(sigma.shape)

  #print(radius)

  B, N, H, W = heatmap.shape

  #data = torch.zeros_like(heatmap)

  #print(data.shape)

  new_data_x = torch.FloatTensor([[-i for i in range(W)]]).to(device)
  new_data_x = torch.unsqueeze(torch.unsqueeze(new_data_x, 0), 0)
  new_data_x = new_data_x.repeat((B, N, H, 1)) #+ torch.unsqueeze(torch.unsqueeze(peak[:,:,1], 2), 3)

  new_data_y = torch.FloatTensor([[-j] for j in range(H)]).to(device)
  new_data_y = torch.unsqueeze(torch.unsqueeze(new_data_y, 0), 0)
  new_data_y = new_data_y.repeat((B, N, 1, W)) #+ torch.unsqueeze(torch.unsqueeze(peak[:,:,0], 2), 3)

  #print(new_data_x[0][0])
  #print("--------")
  #print(new_data_x.shape)
  #print("--------")


  #print(new_data_y[0][0])
  #print("--------")
  #print(new_data_y.shape)
  #print("--------")



  for k in range(0,centers.shape[-1]-1,2):

    #print(torch.unsqueeze(torch.unsqueeze(parameters[:,:,k], 2), 3).shape)

    ##### TO DO : expérimenter avec numpy pour créer mon array de manière bien
    new_data = torch.pow(torch.add(new_data_x,torch.unsqueeze(torch.unsqueeze(centers[:,:,k], 2), 3)), 2) \
               + torch.pow(torch.add(new_data_y,torch.unsqueeze(torch.unsqueeze(centers[:,:,k+1], 2), 3)), 2)

    #print(new_data.shape)
    #print(sigma.shape)

    heatmap += torch.exp(-new_data/(2*sigma**2))

    print(new_data.shape)

    del new_data

  heatmap = (heatmap<=1)*heatmap + (heatmap>1)

  #print(data[0][5])

  if DRAW:
    #max_score = torch.max(src_iou_scores_gaussian,0)
    #print(max_score)
    #print(max_score[1].item())
    ax = sns.heatmap(heatmap.detach().cpu().numpy()[0][0])
    plt.show()

  del new_data_x
  del new_data_y
  del sigma

  return heatmap


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


class PolyLoss(nn.Module):
    def __init__(self, opt):
        super(PolyLoss, self).__init__()
        self.opt = opt

    def forward(self, output, mask, ind, target, freq_mask, hm = None):
        """
        Parameters:
            output: output of polygon head
              [batch_size, 2*nb_vertices, height, width]
            mask: selected objects
              [batch_size, nb_max_objects]
            ind:
              [batch_size, nb_max_objects]
            target: ground-truth for the polygons
              [batch_size, nb_max_objects, 2*nb_vertices]
            hm: output of heatmap head
              [batch_size, nb_categories, height, width]
        Returns:
            loss: scalar
        """

        #print('output', output.shape)
        #print('target', target.shape)

        device = mask.device

        pred = _transpose_and_gather_feat(output, ind)

        #print('pred', pred.shape)

        #predictions = False

        loss = 0.0
        loss_reg = 0.0
        loss_order = 0.0

        sum = 0

        for batch in range(output.shape[0]):

            for i in range(0, pred[batch].shape[0]):  # nbr objects

                if mask[batch][i]:

                    #print(pred[batch][i])

                    sum+=1

                    #predictions = True

                    polygon_mask_pred, polygon_mask_gt = create_mask(output, pred, target, batch, i, self.opt.rep)

                    if self.opt.poly_loss == 'bce':
                        loss += F.binary_cross_entropy(polygon_mask_pred, polygon_mask_gt, reduction='sum')
                    elif self.opt.poly_loss == 'iou' or self.opt.poly_loss == 'l1+iou' or self.opt.poly_loss == 'relu':
                        #print("iou object")
                        intersection = torch.sum((polygon_mask_pred + polygon_mask_gt) == 510)
                        union = torch.sum(polygon_mask_pred != 0) + torch.sum(polygon_mask_gt != 0) - intersection
                        loss += intersection/(union+ 1e-6)

                        #Gradient augmentation for IoU loss ?
                        #loss += intersection/(union+ 1e-6)*torch.sum(polygon_mask_gt != 0)

                        #print(torch.sum(polygon_mask_pred != 0))
                        #print(union)
                        #print(torch.sum(polygon_mask_gt != 0))
                        #print(intersection/(union+ 1e-6)*torch.sum(polygon_mask_gt != 0))
                        #print(loss)

                    if self.opt.poly_order:
                        angles = pred[batch][i][1::2]

                        zero = False
                        for j in range(0, (pred[batch].shape[1] + 1) //2):
                            if angles[j] > 0 :
                                zero = True
                            if angles[j] < 0 and zero:
                                angles[j] += 2*3.14

                        #print(len(angles))
                        #print(pred[batch][i][1::2])
                        #print(angles)
                        #print(target[batch][i][1::2])
                        #print("--------")

                        for j in range(0, (pred[batch].shape[1] - 1) //2):  # points
                            for k in range(j, (pred[batch].shape[1] + 1) //2):
                                if angles[j]-angles[k] > 0 :
                                    loss_order += angles[j]-angles[k]
                                    #print(j)
                                    #print(k)
                                    #print(angles[j]-angles[k])
                                if angles[j]-angles[k] < -7:
                                    loss_order += angles[k]-angles[j]
                                    #print(j)
                                    #print(k)
                                    #print(angles[j]-angles[k])
                        #if loss_order>0.5:
                        #  print(loss_order)
                        #  print(pred[batch][i])
                        #  print(target[batch][i])
                        #  print("------")

                        #  time.sleep(30)

        #if not predictions: #no centers predicted
        #        loss = 0.0

        #print(mask.sum())
        loss_order /= (10*mask.sum() + 1e-4)

        if self.opt.poly_loss == 'iou' or self.opt.poly_loss == 'l1+iou' or self.opt.poly_loss == 'relu':
            loss = 1 - loss / (mask.sum() + 1e-6)
        if self.opt.poly_loss == 'l1' or self.opt.poly_loss == 'l1+iou' or self.opt.poly_loss == 'relu' :
            mask = mask.unsqueeze(2).expand_as(pred).float()

            if self.opt.poly_loss == 'relu' and self.opt.rep == 'cartesian':
                alpha = 20
                data_array = abs(pred-target)
                data_array *= (data_array >= alpha)
                tgt = torch.zeros_like(data_array, requires_grad=True)
                loss_reg = F.l1_loss(data_array * mask, tgt, reduction='sum')
                #print(loss_reg)

            elif self.opt.rep == 'cartesian' :
                #print("l1")
                loss_reg = F.l1_loss(pred * mask, target * mask, reduction='sum')
            elif self.opt.rep == 'polar' :
                #print("l1 polar")
                mask_angles = torch.FloatTensor([1,0]*(output.shape[1]//2))
                mask_angles =mask_angles.to(device)
                mask_angles = mask_angles.unsqueeze(0).unsqueeze(1).expand_as(pred)

                #WEIGHT_ANGLE = 10
                loss_reg = F.l1_loss(pred * mask*mask_angles, target * mask*mask_angles, reduction='sum')
                #loss +=  WEIGHT_ANGLE * F.l1_loss(pred * mask*(1-mask_angles), target * mask*WEIGHT_ANGLE*(1-mask_angles), reduction='sum')
                loss_reg +=  torch.sum(1 - torch.cos(pred * mask*(1-mask_angles) - target * mask*(1-mask_angles)))
                del mask_angles
            elif self.opt.rep == 'polar_fixed' :
                #print("l1 polar fixed")
                mask_angles = torch.FloatTensor([1,0]*(output.shape[1]//2))
                mask_angles = mask_angles.to(device)
                mask_angles = mask_angles.unsqueeze(0).unsqueeze(1).expand_as(pred)

                #WEIGHT_ANGLE = 10
                loss_reg = F.l1_loss(pred * mask*mask_angles, target * mask*mask_angles, reduction='sum')

            #predictions = True
            loss_reg /= (mask.sum() + 1e-6) #/ (freq_mask.mean() + 1e-4)

        #print("iou ", loss)
        #print("l1 ", loss_reg)
        loss += loss_reg #loss=0 if pure regression loss selected

        if self.opt.poly_order :
            #print("iou ", loss)
            #print("order ", loss_order)
            return loss, loss_order

        #print(loss)
        #print("------------")

        return loss

class DiskLoss(nn.Module):
    def __init__(self, opt):
        super(DiskLoss, self).__init__()
        self.opt = opt

    def forward(self, output, mask, ind, target, freq_mask, hm = None):
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

        #print('disk loss')

        pred = _transpose_and_gather_feat(output, ind)
        SUBPIXEL = 1
        OFFSET_Y = SUBPIXEL*output.shape[-1]//4 ###
        OFFSET_X = SUBPIXEL*output.shape[-2]//4 ###

        loss = 0.0
        loss_repulsion = 0.0

        for batch in range(output.shape[0]):

              for i in range(0, pred[batch].shape[0]):  # nbr objects

                  if mask[batch][i]:

                      polygon_mask_pred, polygon_mask_gt = create_mask(output, pred, target, batch, i, self.opt.rep)

                      pred_disks = Image.new('L', (SUBPIXEL*output.shape[-1], SUBPIXEL*output.shape[-2]), 0) ###

                      #print(pred[batch][i])

                      r = pred[batch][i][-1]
                      #loss += (abs(r) - r)/2
                      #print((abs(r) - r)/2)
                      r = math.ceil(abs(r))
                      #print("Le rayon est ", r)



                      for j in range(0, pred[batch].shape[1] - 3, 2):  # points


                      # Test avec 2 disques
                      #for j in range(0, 4, 2):  # points

                          x = pred[batch][i][j]
                          y = pred[batch][i][j+1]

                          """


                          #Force de répulsion
                          for p in range(j+2, pred[batch].shape[1] - 3, 2):  # points

                          # Test avec 2 disques
                          #for p in range(j+2, 4, 2):  # points
                              dist = math.sqrt((x- pred[batch][i][p])*(x- pred[batch][i][p])+(y- pred[batch][i][p+1])*(y- pred[batch][i][p+1]))
                              if dist < 3*r/4:
                                loss_repulsion += 3*r/4- dist
                                #print("r", r)
                                #print("dist", dist)
                                #print("repulse !", 3*r/4- dist)

                          """


                          ImageDraw.Draw(pred_disks).ellipse([(x-r + OFFSET_X, y-r+ OFFSET_Y), (x+r+ OFFSET_X, y+r+ OFFSET_Y)], outline=255, fill=255)

                      pred_disks_tensor = torch.Tensor(np.array(pred_disks)).cuda()
                      #pred_disks.show()


                      intersection = torch.sum((pred_disks_tensor + polygon_mask_gt) == 510)
                      union = torch.sum(pred_disks_tensor != 0) + torch.sum(polygon_mask_gt != 0) - intersection

                      loss += 1 - intersection/(union+ 1e-6)

                      #print(intersection/(union+ 1e-6))

                      #Gradient augmentation
                      #loss += (1-intersection/(union+ 1e-6))*torch.sum(polygon_mask_gt != 0)

                      #print("Aire ", torch.sum(polygon_mask_gt != 0))
                      #print(intersection/(union+ 1e-6))#*torch.sum(polygon_mask_gt != 0))
                      #time.sleep(5)



        loss = loss / (mask.sum() + 1e-6)

        loss_repulsion = loss_repulsion/(mask.sum() + 1e-6)

        return loss, loss_repulsion

class GaussianLoss(nn.Module):
    def __init__(self, opt):
        super(GaussianLoss, self).__init__()
        self.opt = opt

        self.bce = torch.nn.BCELoss(reduction='mean')

    def forward(self, centers, radius, mask, ind, target, peak):
        """
        Parameters:
            centers: output of gaussian centers head
              [batch_size, 2*nb_vertices, height, width]
            radius: output of gaussian std head
              [batch_size, 1, height, width]
            mask: selected objects
              [batch_size, nb_max_objects]
            ind: peak of heatmap (encoded)
              [batch_size, nb_max_objects]
            target: ground-truth for the segmentation mask
              [batch_size, nb_max_objects, height, width]
            peak: ground-truth for the peaks of heatmap
              [batch_size, nb_max_objects, 2]
        Returns:
            loss: scalar
        """        

        #print('peak', peak.shape)
        #print('centers', centers.shape)
        #print('radius', radius.shape)
        #print('target', target.shape)


        pred = _transpose_and_gather_feat(centers, ind)
        pred_radius = _transpose_and_gather_feat(radius, ind)#+10 #.detach().cpu().numpy() +10
        #pred_radius = np.squeeze(pred_radius)


        # Recenter
        pred[:,:,0::2]+=torch.unsqueeze(peak[:,:,0],2)
        pred[:,:,1::2]+=torch.unsqueeze(peak[:,:,1],2)

        #pred_gaussian_tensor = torch.zeros_like(target)
        #pred_gaussian_tensor = display_gaussian_image(pred_gaussian_tensor, pred, pred_radius)#, peak)

        #print(pred_gaussian_tensor.shape)
        #print(pred_radius)

        H, W = target[0][0].shape

        pred = differentiable_gaussian(H,W, pred, pred_radius, self.opt.gaussian_ceiling)


        """
        for batch in range(target.shape[0]):

              for i in range(0, pred[batch].shape[0]):  # nbr objects

                  if mask[batch][i]:

                      pred_gaussian_tensor = torch.zeros_like(target[batch][i], dtype=torch.float)

                      pred_gaussian_tensor = individual_gaussian(pred_gaussian_tensor, pred[batch][i], pred_radius[batch][i])

                      #print(pred[batch][i])

                      pred_gaussian_tensor = (pred_gaussian_tensor<=1)*pred_gaussian_tensor + (pred_gaussian_tensor>1)


                      #ax = sns.heatmap(pred_gaussian_tensor.detach().cpu().numpy())
                      #plt.show()

                      #print(len(pred[batch][i]))

                      #for k in range(0,len(pred[batch][i]),2):
                      #  print(k)
                      #  pred_gaussian_tensor = draw_ellipse_gaussian(pred_gaussian_tensor,
                      #                                                         pred[batch][i][k:k+2], pred_radius[batch][i],pred_radius[batch][i])

                      #  ax = sns.heatmap(pred_gaussian_tensor)
                      #  plt.show()

                      #DRAW = True

                      if DRAW:
                        data = pred_gaussian_tensor.detach().cpu().numpy()

                        ax = sns.heatmap(data)
                        plt.show()

                        ax = sns.heatmap(target[batch][i].detach().cpu().numpy())
                        #print('plot complete data')
                        plt.show()

                      intersection = torch.sum(((pred_gaussian_tensor>self.opt.threshold) + target[batch][i]) == 2)
                      union = torch.sum(pred_gaussian_tensor>self.opt.threshold) + torch.sum(target[batch][i] != 0) - intersection

                      iou_loss += 1 - intersection/(union+ 1e-6)


                      print('differentiable' ,pred_gaussian_tensor.grad_fn)

                      bce_loss += F.binary_cross_entropy_with_logits(pred_gaussian_tensor, target[batch][i], reduction='mean')


                      #print(intersection/(union+ 1e-6))

                      #Gradient augmentation
                      #loss += (1-intersection/(union+ 1e-6))*torch.sum(polygon_mask_gt != 0)

                      #print("Aire ", torch.sum(polygon_mask_gt != 0))
                      #print(intersection/(union+ 1e-6))#*torch.sum(polygon_mask_gt != 0))
                      #time.sleep(5)


        """
        #iou_loss = iou_loss / (mask.sum() + 1e-6)

        #print(pred.shape)
        #print(target.shape)
        #print(mask.shape)

        mask = mask.unsqueeze(-1).unsqueeze(-1)
        # batch_size, nb_max_obj, H, W, nb_points, 2
        mask = mask.expand(mask.shape[0], mask.shape[1], H, W)

        #bce_loss += F.binary_cross_entropy_with_logits(pred*mask, target*mask, reduction='mean')
        #bce_loss = bce_loss/(mask.sum() + 1e-6)

        if self.opt.gaussian_loss == 'bce':
          loss = self.bce(pred*mask, target*mask)
        elif self.opt.gaussian_loss == 'dice':

          inputs = (pred*mask).view(-1)
          targets = (target*mask).view(-1)

          intersection = (inputs * targets).sum()
          dice = (2.*intersection)/(inputs.sum() + targets.sum() + 1e-9)
          loss = 1 - dice
        else :
          raise(NotImplementedError)

        return loss, loss


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

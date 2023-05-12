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
import warnings

DRAW = False

def area(poly_tensor):
  nb_points = poly_tensor.shape[0]

  POLAR = True

  poly_tensor_polar = poly_tensor.clone()

  if POLAR:
    poly_tensor_polar[:,0], poly_tensor_polar[:,1] = torch.mul(poly_tensor[:,0], torch.cos(poly_tensor[:,1])), torch.mul(poly_tensor[:,0], torch.sin(poly_tensor[:,1]))
  #print(poly_tensor.shape)
  double = poly_tensor_polar.repeat((2,1))

  #print(double)
  polyleft = torch.mul(double[0:nb_points+1,0],double[1:nb_points+2,1])
  polyright = torch.mul(double[0:nb_points+1,1],double[1:nb_points+2,0])

  return torch.abs(0.5*(torch.sum(polyright)-torch.sum(polyleft)))

def is_convex_polygon(polygon):
    """Return True if the polynomial defined by the sequence of 2D
    points is 'strictly convex': points are valid, side lengths non-
    zero, interior angles are strictly between zero and a straight
    angle, and the polygon does not intersect itself.

    NOTES:  1.  Algorithm: the signed changes of the direction angles
                from one side to the next side must be all positive or
                all negative, and their sum must equal plus-or-minus
                one full turn (2 pi radians). Also check for too few,
                invalid, or repeated points.
            2.  No check is explicitly done for zero internal angles
                (180 degree direction-change angle) as this is covered
                in other ways, including the `n < 3` check.
    """
    try:  # needed for any bad points or direction changes
        # Check for too few points
        if len(polygon) < 3:
            return False
        # Get starting information
        old_x, old_y = polygon[-2]
        new_x, new_y = polygon[-1]
        new_direction = atan2(new_y - old_y, new_x - old_x)
        angle_sum = 0.0
        # Check each point (the side ending there, its angle) and accum. angles
        for ndx, newpoint in enumerate(polygon):
            # Update point coordinates and side directions, check side length
            old_x, old_y, old_direction = new_x, new_y, new_direction
            new_x, new_y = newpoint
            new_direction = math.atan2(new_y - old_y, new_x - old_x)
            if old_x == new_x and old_y == new_y:
                return False  # repeated consecutive points
            # Calculate & check the normalized direction-change angle
            angle = new_direction - old_direction
            if angle <= -pi:
                angle += TWO_PI  # make it in half-open interval (-Pi, Pi]
            elif angle > pi:
                angle -= TWO_PI
            if ndx == 0:  # if first time through loop, initialize orientation
                if angle == 0.0:
                    return False
                orientation = 1.0 if angle > 0.0 else -1.0
            else:  # if other time through loop, check orientation is stable
                if orientation * angle <= 0.0:  # not both pos. or both neg.
                    return False
            # Accumulate the direction-change angle
            angle_sum += angle
        # Check that the total number of full turns is plus-or-minus 1
        return abs(round(angle_sum / TWO_PI)) == 1
    except (ArithmeticError, TypeError, ValueError):
        return False  # any exception means not a proper convex polygon

def convex_hull_graham(points):
    '''
    Returns points on convex hull in CCW order according to Graham's scan algorithm.
    By Tom Switzer thomas.switzer@gmail.com.
    '''
    TURN_LEFT, TURN_RIGHT, TURN_NONE = (1, -1, 0)

    def cmp(a, b):
        return (a > b) - (a < b)

    def turn(p, q, r):
        return cmp((q[0] - p[0])*(r[1] - p[1]) - (r[0] - p[0])*(q[1] - p[1]), 0)

    def _keep_left(hull, r):
        while len(hull) > 1 and turn(hull[-2], hull[-1], r) != TURN_LEFT:
            hull.pop()
        if not len(hull) or hull[-1] != r:
            hull.append(r)
        return hull

    points = sorted(points)

    l = reduce(_keep_left, points, [])
    u = reduce(_keep_left, reversed(points), [])
    return l.extend(u[i] for i in range(1, len(u) - 1)) or l

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

def is_convex(polygon):
    zcrossproduct_A = 1
    zcrossproduct_B = 1

    # iterate over each vertex in the polygon 
    for i in range(len(polygon)): 
        p1 = polygon[i-1] 
        p2 = polygon[i] 
        p3 = polygon[(i+1) % len(polygon)] 
 
        # compute cross-product of each triplet
        dx1 = p2[0]-p1[0]
        dy1 = p2[1]-p1[1]
        dx2 = p3[0]-p2[0]
        dy2 = p3[1]-p2[1]
        cross = dx1*dy2 - dy1*dx2
        zcrossproduct_A *= cross + torch.abs(cross)
        zcrossproduct_B *= torch.abs(cross) - cross
    
    return zcrossproduct_A != 0.0 or zcrossproduct_B != 0.0

def isInTriangle(p1, p2, p3, x):
    denominator = ((p2[1]-p3[1])*(p1[0]-p3[0])+(p3[0]-p2[0])*(p1[1]-p3[1]))  
    a = ((p2[1]-p3[1])*(x[0]-p3[0])+(p3[0]-p2[0])*(x[1]-p3[1]))/denominator
    b = ((p3[1]-p1[1])*(x[0]-p3[0])+(p1[0]-p3[0])*(x[1]-p3[1]))/denominator
    c = 1 - a - b

    return 0 <= a and a <= 1 and 0 <= b and b <= 1 and 0 <= c and c <= 1

def is_ear(p1, p2, p3, polygon, i):
    boolean = True
    for p in range(len(polygon)):
      if p != i and p != (i+1) % len(polygon) and p != (i-1) % len(polygon):
        boolean= boolean and not isInTriangle(p1, p2, p3, polygon[p])
    return boolean

def find_ear(polygon, poly_tensor_polar): 
    # initialize the output ear 
    ear = None 
 
    # iterate over each vertex in the polygon 
    for i in range(len(polygon)): 
        p1 = poly_tensor_polar[i-1] 
        p2 = poly_tensor_polar[i] 
        p3 = poly_tensor_polar[(i+1) % len(polygon)] 

        #print('new', p1)
 
        # check if the vertex is an ear 
        if is_ear(p1, p2, p3, poly_tensor_polar, i): 
            #print(p1)
            return torch.cat((torch.cat((polygon[i-1], polygon[i])), polygon[(i+1) % len(polygon)])).view(-1,2), i

def ear_clipping(polygon, poly_tensor_polar): 
    # initialize the output list of convex subpolygons 
    convex_polygons = [] 
    # loop until all ears are clipped 
    while len(polygon) > 3: 
        # find an ear 
        try :
            ear, i = find_ear(polygon) 
            # clip the ear 
            
            #polygon = polygon[:i]+ polygon[(i+1) % len(polygon):]
            convex_polygons.append(ear)

            if i == 0 or i == len(polygon)-1:
                polygon = polygon[1:]
            elif i == len(polygon)-1:
                polygon = polygon[:i]
            else:
                polygon = np.vstack((polygon[:i], polygon[(i+1) % len(polygon):]))
        except TypeError: 
            break        
    convex_polygons.append(polygon)
    return convex_polygons

def divide_concave_polygon(polygon): 
 
    # check if the polygon is already convex 
    if is_convex(polygon): 
        return[polygon] 
    else: 
        # use the Ear Clipping algorithm to divide the polygon into convex subpolygons 
      POLAR = True

      poly_tensor_polar = polygon.clone()

      if POLAR:
          poly_tensor_polar = torch.cat((torch.mul(polygon[:,0], torch.cos(polygon[:,1])).view(-1,1), torch.mul(polygon[:,0], torch.sin(polygon[:,1])).view(-1,1)), 1)
          #poly_tensor_polar[:,0], poly_tensor_polar[:,1] = torch.mul(polygon[:,0], torch.cos(polygon[:,1])), torch.mul(polygon[:,0], torch.sin(polygon[:,1]))
          
      return ear_clipping(polygon, poly_tensor_polar)

class WeilPolygonClipper:
    
    def __init__(self,warn_if_empty=True):
        self.warn_if_empty = warn_if_empty
    
    def is_inside(self,c1,c2,c):

        POLAR = True

        if POLAR:
          p1 = c1[0]*torch.cos(c1[1]), c1[0]*torch.sin(c1[1])
          p2 = c2[0]*torch.cos(c2[1]), c2[0]*torch.sin(c2[1])
          q = c[0]*torch.cos(c[1]), c[0]*torch.sin(c[1])
          #R = (p2[0]*torch.cos(p2[1]) - p1[0]*torch.cos(p1[1])) * (q[0]*torch.sin(q[1]) - p1[0]*torch.sin(p1[1])) - (p2[0]*torch.sin(p2[1]) - p1[0]*torch.sin(p1[1])) * (q[0]*torch.cos(q[1]) - p1[0]*torch.cos(p1[1]))

        else :
          p1, p2, q = c1, c2, c
        R = (p2[0] - p1[0]) * (q[1] - p1[1]) - (p2[1] - p1[1]) * (q[0] - p1[0])

        if R < 0:
            return 1
        elif R ==0:
            return 2
        else :
            return 0

        #return (R <= 0)
    
    def compute_intersection(self,c1,c2,c3,c4):
        
        """
        given points p1 and p2 on line L1, compute the equation of L1 in the
        format of y = m1 * x + b1. Also, given points p3 and p4 on line L2,
        compute the equation of L2 in the format of y = m2 * x + b2.
        
        To compute the point of intersection of the two lines, equate
        the two line equations together
        
        m1 * x + b1 = m2 * x + b2
        
        and solve for x. Once x is obtained, substitute it into one of the
        equations to obtain the value of y.    
        
        if one of the lines is vertical, then the x-coordinate of the point of
        intersection will be the x-coordinate of the vertical line. Note that
        there is no need to check if both lines are vertical (parallel), since
        this function is only called if we know that the lines intersect.
        """
        POLAR = True

        if POLAR:
          p1 = c1[0]*torch.cos(c1[1]), c1[0]*torch.sin(c1[1])
          p2 = c2[0]*torch.cos(c2[1]), c2[0]*torch.sin(c2[1])
          p3 = c3[0]*torch.cos(c3[1]), c3[0]*torch.sin(c3[1])
          p4 = c4[0]*torch.cos(c4[1]), c4[0]*torch.sin(c4[1])

        else :
          p1, p2, p3, p4 = c1, c2, c3,c4
        
        # if first line is vertical
        if p2[0] - p1[0] == 0:
            x = p1[0]
            
            # slope and intercept of second line
            m2 = (p4[1] - p3[1]) / (p4[0] - p3[0])
            b2 = p3[1] - m2 * p3[0]
            
            # y-coordinate of intersection
            y = m2 * x + b2
        
        # if second line is vertical
        elif p4[0] - p3[0] == 0:
            x = p3[0]
            
            # slope and intercept of first line
            m1 = (p2[1] - p1[1]) / (p2[0] - p1[0])
            b1 = p1[1] - m1 * p1[0]
            
            # y-coordinate of intersection
            y = m1 * x + b1
        
        # if neither line is vertical
        else:
            m1 = (p2[1] - p1[1]) / (p2[0] - p1[0])
            b1 = p1[1] - m1 * p1[0]
            
            # slope and intercept of second line
            m2 = (p4[1] - p3[1]) / (p4[0] - p3[0])
            b2 = p3[1] - m2 * p3[0]
        
            # x-coordinate of intersection
            x = (b2 - b1) / (m1 - m2)
        
            # y-coordinate of intersection
            y = m1 * x + b1

        if POLAR:
          r = torch.sqrt(x*x+y*y)
          theta = torch.atan((y+1e-8)/(x+1e-8))

          if x < 0:
            theta = theta + math.pi
          elif y < 0: 
            theta = theta + 2*math.pi

          # need to unsqueeze so torch.cat doesn't complain outside func
          intersection = torch.stack((r,theta)).unsqueeze(0)

        else:
      
          # need to unsqueeze so torch.cat doesn't complain outside func
          intersection = torch.stack((x,y)).unsqueeze(0)
        
        return intersection
    
    def clip(self,subject_polygon,clipping_polygon):
        # it is assumed that requires_grad = True only for clipping_polygon
        # subject_polygon and clipping_polygon are N x 2 and M x 2 torch
        # tensors respectively

        device = clipping_polygon.device

        final_polygon = torch.empty((0,2)).to(device)

        #subject_polygon, indices = torch.sort(subject_polygon, 0)


        inters = torch.empty((0,2)).to(device)

        # list of places for intersections (first subject then clipping then intersection)
        inbounds = []
        outbounds = []

        for i in range(len(clipping_polygon)):

            c_edge_start = clipping_polygon[i - 1]
            c_edge_end = clipping_polygon[i]

            #print('tgt', c_edge_start, c_edge_end)

            for j in range(len(subject_polygon)):

                s_edge_start = subject_polygon[j - 1]
                s_edge_end = subject_polygon[j]

                to_end = self.is_inside(c_edge_start,c_edge_end,s_edge_end)
                to_start = self.is_inside(c_edge_start,c_edge_end,s_edge_start)

                #print('pred', s_edge_start, s_edge_end)
              
                if to_end == 1:
                    #print('s_end inside c')
                    if to_start == 0:
                        #print('s_start outside c')
                        # Test actual intersection
                        c_in_start, c_in_end = self.is_inside(s_edge_start,s_edge_end,c_edge_end), self.is_inside(s_edge_start,s_edge_end,c_edge_start)
                        if c_in_start !=  c_in_end :
                          #print('intersect')
                          intersection = self.compute_intersection(s_edge_start,s_edge_end,c_edge_start,c_edge_end)
                          inters = torch.cat((inters,intersection),dim=0)
                          outbounds.append([j,i, len(inters)-1])

                elif to_start == 1:
                    #print('s_start inside c')
                    if to_end == 0:
                        #print('s_end outside c')
                        # Test actual intersection
                        c_in_start, c_in_end = self.is_inside(s_edge_start,s_edge_end,c_edge_end), self.is_inside(s_edge_start,s_edge_end,c_edge_start)
                        if c_in_start !=  c_in_end :
                          #print('intersect')
                          intersection = self.compute_intersection(s_edge_start,s_edge_end,c_edge_start,c_edge_end)
                          inters = torch.cat((inters,intersection),dim=0)
                          inbounds.append([j,i, len(inters)-1])

                elif to_end == 2:
                  if to_start == 0:
                    c_in_start, c_in_end = self.is_inside(s_edge_start,s_edge_end,c_edge_end), self.is_inside(s_edge_start,s_edge_end,c_edge_start)
                    if c_in_start !=  c_in_end :
                          #print('intersect')
                          intersection = self.compute_intersection(s_edge_start,s_edge_end,c_edge_start,c_edge_end)
                          inters = torch.cat((inters,intersection),dim=0)
                          outbounds.append([j,i, len(inters)-1])

                  elif to_start == 1:
                    c_in_start, c_in_end = self.is_inside(s_edge_start,s_edge_end,c_edge_end), self.is_inside(s_edge_start,s_edge_end,c_edge_start)
                    if c_in_start !=  c_in_end :
                          #print('intersect')
                          intersection = self.compute_intersection(s_edge_start,s_edge_end,c_edge_start,c_edge_end)
                          inters = torch.cat((inters,intersection),dim=0)
                          inbounds.append([j,i, len(inters)-1])
                    
                elif to_start == 2:
                  if to_end == 0:
                    c_in_start, c_in_end = self.is_inside(s_edge_start,s_edge_end,c_edge_end), self.is_inside(s_edge_start,s_edge_end,c_edge_start)
                    if c_in_start !=  c_in_end :
                          #print('intersect')
                          intersection = self.compute_intersection(s_edge_start,s_edge_end,c_edge_start,c_edge_end)
                          inters = torch.cat((inters,intersection),dim=0)
                          inbounds.append([j,i, len(inters)-1])
                    
                  elif to_end == 1:
                    c_in_start, c_in_end = self.is_inside(s_edge_start,s_edge_end,c_edge_end), self.is_inside(s_edge_start,s_edge_end,c_edge_start)
                    if c_in_start !=  c_in_end :
                          #print('intersect')
                          intersection = self.compute_intersection(s_edge_start,s_edge_end,c_edge_start,c_edge_end)
                          inters = torch.cat((inters,intersection),dim=0)
                          outbounds.append([j,i, len(inters)-1])

        outbounds = np.array(outbounds)
        inbounds = np.array(inbounds)

       # print(outbounds)
       # print(inbounds)
       # print(inters)

        used_inters = []

        while len(used_inters) < len(inbounds) :
          new_inter = 0
          stop = inbounds[new_inter][0:2]
          j = inbounds[new_inter][0]# j -> là où on en est dans le polygone sujet
          i = inbounds[new_inter][1]# i -> là où on en est dans le polygone clipping

          start = True

          # Tant qu'on a pas retrouvé l'intersection de départ (aka la coord de l'intersection dans le poly sujet)
          while j != stop[0] or i != stop[1] or start:
            start = False
            # Tant qu'on atteint pas une intersection out (aka la coord de l'intersection dans )
            while j not in outbounds[:,0]:
              final_polygon = torch.cat((final_polygon, subject_polygon[j].unsqueeze(0)),dim=0)
              j= (j+1)%len(subject_polygon)

            new_inter = np.where(outbounds[:,0]==j)[0][0]
            final_polygon = torch.cat((final_polygon, inters[outbounds[new_inter][2]].unsqueeze(0)),dim=0)
            i = outbounds[new_inter][1]
            
            while i not in inbounds[:,1]:
              final_polygon = torch.cat((final_polygon, clipping_polygon[i].unsqueeze(0)),dim=0)
              i= (i+1)%len(clipping_polygon)
            new_inter = np.where(inbounds[:,1] == i)[0][0]
            j = inbounds[new_inter][0]
            final_polygon = torch.cat((final_polygon, inters[inbounds[new_inter][2]].unsqueeze(0)),dim=0)
            inbounds = np.vstack((inbounds[:new_inter], inbounds[new_inter +1:]))
            used_inters.append(new_inter)


        return final_polygon

          
    def __call__(self,A,B):
        clipped_polygon = self.clip(A,B)
        if len(clipped_polygon) == 0 and self.warn_if_empty:
            warnings.warn("No intersections found. Are you sure your polygon coordinates are in clockwise order?")
        
        return clipped_polygon

class PolygonClipper:
    
    def __init__(self,warn_if_empty=True):
        self.warn_if_empty = warn_if_empty
    
    def is_inside(self,p1,p2,q):

        POLAR = False

        if POLAR:
          R = (p2[0]*torch.cos(p2[1]) - p1[0]*torch.cos(p1[1])) * (q[0]*torch.sin(q[1]) - p1[0]*torch.sin(p1[1])) - (p2[0]*torch.sin(p2[1]) - p1[0]*torch.sin(p1[1])) * (q[0]*torch.cos(q[1]) - p1[0]*torch.cos(p1[1]))

        else :
          R = (p2[0] - p1[0]) * (q[1] - p1[1]) - (p2[1] - p1[1]) * (q[0] - p1[0])
        
        
        if R <= 0:
            return True
        else:
            return False
    
    def compute_intersection(self,c1,c2,c3,c4):
        
        """
        given points p1 and p2 on line L1, compute the equation of L1 in the
        format of y = m1 * x + b1. Also, given points p3 and p4 on line L2,
        compute the equation of L2 in the format of y = m2 * x + b2.
        
        To compute the point of intersection of the two lines, equate
        the two line equations together
        
        m1 * x + b1 = m2 * x + b2
        
        and solve for x. Once x is obtained, substitute it into one of the
        equations to obtain the value of y.    
        
        if one of the lines is vertical, then the x-coordinate of the point of
        intersection will be the x-coordinate of the vertical line. Note that
        there is no need to check if both lines are vertical (parallel), since
        this function is only called if we know that the lines intersect.
        """

        POLAR = False

        if POLAR:
          p1 = c1[0]*torch.cos(c1[1]), c1[0]*torch.sin(c1[1])
          p2 = c2[0]*torch.cos(c2[1]), c2[0]*torch.sin(c2[1])
          p3 = c3[0]*torch.cos(c3[1]), c3[0]*torch.sin(c3[1])
          p4 = c4[0]*torch.cos(c4[1]), c4[0]*torch.sin(c4[1])

        else : 
          p1, p2, p3, p4 = c1, c2, c3, c4

        #print(p1, p2, p3, p4)
        
        # if first line is vertical
        if p2[0] - p1[0] == 0:
            x = p1[0]
            
            # slope and intercept of second line
            m2 = (p4[1] - p3[1]) / (p4[0] - p3[0])
            b2 = p3[1] - m2 * p3[0]
            
            # y-coordinate of intersection
            y = m2 * x + b2
        
        # if second line is vertical
        elif p4[0] - p3[0] == 0:
            x = p3[0]
            
            # slope and intercept of first line
            m1 = (p2[1] - p1[1]) / (p2[0] - p1[0])
            b1 = p1[1] - m1 * p1[0]
            
            # y-coordinate of intersection
            y = m1 * x + b1
        
        # if neither line is vertical
        else:
            m1 = (p2[1] - p1[1]) / (p2[0] - p1[0])
            b1 = p1[1] - m1 * p1[0]
            
            # slope and intercept of second line
            m2 = (p4[1] - p3[1]) / (p4[0] - p3[0])
            b2 = p3[1] - m2 * p3[0]
        
            # x-coordinate of intersection
            x = (b2 - b1) / (m1 - m2)
        
            # y-coordinate of intersection
            y = m1 * x + b1

        if POLAR:
          r = torch.sqrt(x*x+y*y)
          theta = torch.atan((y+1e-8)/(x+1e-8))

          if x < 0:
            theta = theta + math.pi
          elif y < 0: 
            theta = theta + 2*math.pi

          # need to unsqueeze so torch.cat doesn't complain outside func
          intersection = torch.stack((r,theta)).unsqueeze(0)

        else:
      
          # need to unsqueeze so torch.cat doesn't complain outside func
          intersection = torch.stack((x,y)).unsqueeze(0)
        
        return intersection
    
    def clip(self,subject_polygon,clipping_polygon):
        # it is assumed that requires_grad = True only for clipping_polygon
        # subject_polygon and clipping_polygon are N x 2 and M x 2 torch
        # tensors respectively
        device = clipping_polygon.device

        final_polygon = torch.clone(subject_polygon).to(device)
        
        for i in range(len(clipping_polygon)):
            
            # stores the vertices of the next iteration of the clipping procedure
            # final_polygon consists of list of 1 x 2 tensors 
            next_polygon = torch.clone(final_polygon).to(device)
            
            # stores the vertices of the final clipped polygon. This will be
            # a K x 2 tensor, so need to initialize shape to match this
            final_polygon = torch.empty((0,2)).to(device)
            
            # these two vertices define a line segment (edge) in the clipping
            # polygon. It is assumed that indices wrap around, such that if
            # i = 0, then i - 1 = M.
            c_edge_start = clipping_polygon[i - 1]
            c_edge_end = clipping_polygon[i]
            
            for j in range(len(next_polygon)):
                
                # these two vertices define a line segment (edge) in the subject
                # polygon
                s_edge_start = next_polygon[j - 1]
                s_edge_end = next_polygon[j]
                
                if self.is_inside(c_edge_start,c_edge_end,s_edge_end):
                    if not self.is_inside(c_edge_start,c_edge_end,s_edge_start):
                        intersection = self.compute_intersection(s_edge_start,s_edge_end,c_edge_start,c_edge_end)
                        final_polygon = torch.cat((final_polygon,intersection),dim=0)
                    final_polygon = torch.cat((final_polygon,s_edge_end.unsqueeze(0)),dim=0)
                elif self.is_inside(c_edge_start,c_edge_end,s_edge_start):
                    intersection = self.compute_intersection(s_edge_start,s_edge_end,c_edge_start,c_edge_end)
                    final_polygon = torch.cat((final_polygon,intersection),dim=0)
        
        return final_polygon
    
    def __call__(self,A,B):
        clipped_polygon = self.clip(A,B)
        if len(clipped_polygon) == 0 and self.warn_if_empty:
            warnings.warn("No intersections found. Are you sure your \
                          polygon coordinates are in clockwise order?")
        
        return clipped_polygon


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

    def forward(self, output, mask, ind, target, freq_mask = None, peak = None, hm = None):
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

        device = mask.device

        pred = _transpose_and_gather_feat(output, ind)

        loss = 0.0
        loss_reg = 0.0
        loss_order = 0.0

        sum = 0

        clip = WeilPolygonClipper()
        #clip = PolygonClipper()

        for batch in range(output.shape[0]):

            for i in range(0, pred[batch].shape[0]):  # nbr objects

                if mask[batch][i]:

                    sum+=1

                    if self.opt.poly_loss == 'iou' or self.opt.poly_loss == 'l1+iou' or self.opt.poly_loss == 'relu':

                        #How to sort: get indices when sorting angle columns and apply it to the whole tensor
                        sorted_pred = pred[batch][i].view(-1,2)[torch.sort(pred[batch][i].view(-1,2)[:,1],0)[1]]
                        sorted_pred = torch.cat((torch.abs(sorted_pred[:,0]).unsqueeze(1), sorted_pred[:,1].unsqueeze(1)), 1)

                        clipped_polygon = clip(sorted_pred, target[batch][i].view(-1,2)) #.flip(0))

                        area_intersection = area(clipped_polygon)
                        intersection = (area_intersection.item() == 0.0)*torch.min(area(sorted_pred),area(target[batch][i].view(-1,2)))+area_intersection
                        union = area(target[batch][i].view(-1,2)) + area(sorted_pred) - intersection

                        loss += intersection/(union+ 1e-6)


                    if self.opt.poly_order:
                        angles = pred[batch][i][1::2]

                        zero = False
                        for j in range(0, (pred[batch].shape[1] + 1) //2):
                            if angles[j] > 0 :
                                zero = True
                            if angles[j] < 0 and zero:
                                angles[j] += 2*3.14

                        for j in range(0, (pred[batch].shape[1] - 1) //2):  # points
                            for k in range(j, (pred[batch].shape[1] + 1) //2):
                                if angles[j]-angles[k] > 0 :
                                    loss_order += angles[j]-angles[k]

        loss_order /= (10*mask.sum() + 1e-4)

        if self.opt.poly_loss == 'iou' or self.opt.poly_loss == 'l1+iou' or self.opt.poly_loss == 'relu':
            loss = (1 - loss / (mask.sum() + 1e-6))
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

        # print("iou ", loss)
        # print("l1 ", loss_reg)
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

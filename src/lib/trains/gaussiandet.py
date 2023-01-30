from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np

from models.losses import FocalLoss
from models.losses import RegL1Loss, RegLoss, GaussianLoss, NormRegL1Loss, RegWeightedL1Loss, AreaPolyLoss
from models.decode import gaussiandet_decode
from models.utils import _sigmoid
from utils.debugger import Debugger
from utils.post_process import gaussiandet_post_process
from utils.oracle_utils import gen_oracle_map
from .base_trainer import BaseTrainer
import cv2
from utils.image import get_affine_transform, affine_transform

class GaussiandetLoss(torch.nn.Module):
    def __init__(self, opt):
        super(GaussiandetLoss, self).__init__()
        self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
        self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
            RegLoss() if opt.reg_loss == 'sl1' else None

        self.crit_gaussian = GaussianLoss(opt)

        self.crit_wh = torch.nn.L1Loss(reduction='sum') if opt.dense_wh else \
            NormRegL1Loss() if opt.norm_wh else \
                RegWeightedL1Loss() if opt.cat_spec_wh else self.crit_reg
        self.opt = opt

    def forward(self, outputs, batch):
        #print(batch.keys())
        opt = self.opt
        hm_loss, off_loss, poly_loss, depth_loss, wh_loss, fg_loss, dice_loss, bce_loss = 0, 0, 0, 0, 0, 0, 0, 0
        for s in range(opt.num_stacks):
            output = outputs[s]
            if not opt.mse_loss:
                output['hm'] = _sigmoid(output['hm'])
                # output['fg'] = _sigmoid(output['fg'])
                # output['border_hm'] = _sigmoid(output['border_hm'])

            depth_loss += self.crit_reg(output['pseudo_depth'], batch['reg_mask'],
                                          batch['ind'], batch['pseudo_depth']) / opt.num_stacks

            hm_loss += self.crit(output['hm'], batch['hm']) / opt.num_stacks

            dice, bce = self.crit_gaussian(output['centers'], output['radius'], batch[
                'reg_mask'], batch['ind'], batch['masks'], batch['peak'])
            dice_loss += dice / opt.num_stacks
            bce_loss += bce / opt.num_stacks

            if opt.reg_offset and opt.off_weight > 0:
                off_loss += self.crit_reg(output['reg'], batch['reg_mask'],
                                          batch['ind'], batch['reg']) / opt.num_stacks

        # import cv2
        # import os
        # write_depth = np.array(output['fg'][0, :, :, :].cpu().detach().squeeze(0).squeeze(0))
        # # print(write_depth.shape)
        # write_depth = (((write_depth - np.min(write_depth)) / np.max(write_depth)) * 255).astype(np.uint8)
        # count = 0
        # write_name = '/store/datasets/cityscapes/test_images/depth/depth' + str(count) + '.jpg'
        # while os.path.exists(write_name):
        #     count += 1
        #     write_name = '/store/datasets/cityscapes/test_images/depth/depth' + str(count) + '.jpg'
        # cv2.imwrite('/store/datasets/cityscapes/test_images/depth/depth' + str(count) + '.jpg', write_depth)
        # exit()

        # loss = opt.hm_weight * hm_loss + opt.off_weight * off_loss + opt.poly_weight * poly_loss + opt.depth_weight * depth_loss
        loss = opt.hm_weight * hm_loss + opt.off_weight * off_loss + opt.poly_weight * poly_loss \
               + opt.depth_weight * depth_loss # + fg_loss #  + opt.wh_weight * wh_loss #  + opt.border_hm_weight * border_hm_loss
        # loss_stats = {'loss': loss, 'hm_loss': hm_loss, 'off_loss': off_loss, 'poly_loss': poly_loss, 'depth_loss': depth_loss, 'border_hm_loss': border_hm_loss}

        #loss = opt.hm_weight * hm_loss + opt.off_weight * off_loss + opt.poly_weight * (dice_loss + bce_loss) \
        #       + opt.depth_weight * depth_loss #
        loss = opt.hm_weight * hm_loss + opt.off_weight * off_loss + opt.poly_weight * bce_loss \
               + opt.depth_weight * depth_loss ##  + fg_loss #  + opt.wh_weight * wh_loss #  + opt.border_hm_weight * border_hm_loss

        loss_stats = {'loss': loss, 'hm_l': hm_loss, 'off_l': off_loss, 'bce_l': bce_loss, 'iou_l': dice_loss,
                      'depth_l': depth_loss}

        #print(loss_stats)

        return loss, loss_stats


class GaussianTrainer(BaseTrainer):
    def __init__(self, opt, model, optimizer=None):
        super(GaussianTrainer, self).__init__(opt, model, optimizer=optimizer)

    def _get_losses(self, opt):
        loss_states = ['loss', 'hm_l', 'off_l', 'bce_l', 'iou_l', 'depth_l']
        loss = GaussiandetLoss(opt)
        return loss_states, loss

    def debug(self, batch, output, iter_id):
        opt = self.opt
        reg = output['reg'] if opt.reg_offset else None
        dets = gaussiandet_decode(
            output['hm'], output['centers'], output['radius'], output['depth'], reg=reg, K=opt.K)
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        dets[:, :, :4] *= opt.down_ratio
        dets_gt = batch['meta']['gt_det'].numpy().reshape(1, -1, dets.shape[2])
        dets_gt[:, :, :4] *= opt.down_ratio
        for i in range(1):
            debugger = Debugger(
                dataset=opt.dataset, ipynb=(opt.debug == 3), theme=opt.debugger_theme)
            img = batch['input'][i].detach().cpu().numpy().transpose(1, 2, 0)
            img = np.clip(((
                                   img * opt.std + opt.mean) * 255.), 0, 255).astype(np.uint8)
            pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
            gt = debugger.gen_colormap(batch['hm'][i].detach().cpu().numpy())
            debugger.add_blend_img(img, pred, 'pred_hm')
            debugger.add_blend_img(img, gt, 'gt_hm')
            debugger.add_img(img, img_id='out_pred')
            for k in range(len(dets[i])):
                if dets[i, k, 4] > opt.center_thresh:
                    debugger.add_coco_bbox(dets[i, k, :4], dets[i, k, -1],
                                           dets[i, k, 4], img_id='out_pred')

            debugger.add_img(img, img_id='out_gt')
            for k in range(len(dets_gt[i])):
                if dets_gt[i, k, 4] > opt.center_thresh:
                    debugger.add_coco_bbox(dets_gt[i, k, :4], dets_gt[i, k, -1],
                                           dets_gt[i, k, 4], img_id='out_gt')

            if opt.debug == 4:
                debugger.save_all_imgs(opt.debug_dir, prefix='{}'.format(iter_id))
            else:
                debugger.show_all_imgs(pause=True)

    def save_result(self, output, batch, results):
        reg = output['reg'] if self.opt.reg_offset else None
        dets = gaussiandet_decode(
            output['hm'], output['centers'], output['radius'], output['pseudo_depth'], reg=reg, K=self.opt.K)
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        dets_out = gaussiandet_post_process(
            dets.copy(), batch['meta']['c'].cpu().numpy(),
            batch['meta']['s'].cpu().numpy(),
            output['hm'].shape[2], output['hm'].shape[3], output['hm'].shape[1])

        # fg = np.array(output['fg'].cpu().detach().squeeze(0).squeeze(0))
        # if np.max(fg) != 0:
        #     fg = (fg - np.min(fg)) / np.max(fg)
        # dets_out[0]['fg'] = fg
        results[batch['meta']['img_id'].cpu().numpy()[0]] = dets_out[0]
        # results[str(batch['meta']['img_id'].cpu().numpy()[0])+'_fg'] = fg

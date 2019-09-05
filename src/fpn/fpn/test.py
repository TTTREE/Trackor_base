# --------------------------------------------------------
# Pytorch FPN implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Jianwei Yang, based on code from faster R-CNN
# --------------------------------------------------------

from __future__ import absolute_import, division, print_function

import os
import sys
import time
import numpy as np
import torch
from torch.autograd import Variable

import cv2
from fpn.model.nms.nms_wrapper import nms, soft_nms
from fpn.model.rpn.bbox_transform import bbox_transform_inv, clip_boxes
from fpn.model.utils.config import cfg

try:
    xrange  # Python 2
except NameError:
    xrange = range  # Python 3


def validate(fpn, dataloader, imdb, vis=False, cuda=False, soft_nms=False, score_thresh=0.05):
    num_images = len(imdb.image_index)
    max_per_image = 100

    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]

    im_data = Variable(torch.FloatTensor(1), volatile=True)
    im_info = Variable(torch.FloatTensor(1), volatile=True)
    num_boxes = Variable(torch.LongTensor(1), volatile=True)
    gt_boxes = Variable(torch.FloatTensor(1), volatile=True)

    # ship to cuda
    if cuda:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

    fpn.eval()

    data_iter = iter(dataloader)
    empty_array = np.transpose(np.array([[], [], [], [], []]), (1, 0))
    for i in range(num_images):
        data = data_iter.next()
        im_data.data.resize_(data[0].size()).copy_(data[0])
        im_info.data.resize_(data[1].size()).copy_(data[1])
        gt_boxes.data.resize_(data[2].size()).copy_(data[2])
        num_boxes.data.resize_(data[3].size()).copy_(data[3])

        det_tic = time.time()
        rois, cls_prob, bbox_pred, \
        _, _, _, _, _ = fpn(im_data, im_info, gt_boxes, num_boxes)
        # rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_box, \
        #     RCNN_loss_cls, RCNN_loss_bbox, \
        #     _ = fpn(im_data, im_info, gt_boxes, num_boxes)

        # loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
        #         + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
        # # print(rois, cls_prob, bbox_pred)
        # print(loss.data.cpu().numpy())
        # continue

        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]

        if cfg.TEST.BBOX_REG:
            # Apply bounding-box regression deltas
            box_deltas = bbox_pred.data
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                # Optionally normalize targets by a precomputed mean and stdev
                if cfg.CLASS_AGNOSTIC_BBX_REG:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                 + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    box_deltas = box_deltas.view(1, -1, 4)
                else:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                 + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    box_deltas = box_deltas.view(1, -1, 4 * len(imdb.classes))
            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
            pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
        else:
            # Simply repeat the boxes, once for each class
            pred_boxes = boxes

        pred_boxes /= data[1][0][2]

        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()
        det_toc = time.time()
        detect_time = det_toc - det_tic
        misc_tic = time.time()
        if vis:
            im = cv2.imread(imdb.image_path_at(i))
            im2show = np.copy(im)
        for j in xrange(1, imdb.num_classes):
            inds = torch.nonzero(scores[:, j] > score_thresh).view(-1)
            # if there is det
            if inds.numel() > 0:
                cls_scores = scores[:, j][inds]
                _, order = torch.sort(cls_scores, 0, True)
                if cfg.CLASS_AGNOSTIC_BBX_REG:
                    cls_boxes = pred_boxes[inds, :]
                else:
                    cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                cls_dets = cls_dets[order]
                if soft_nms:
                    np_dets = cls_dets.cpu().numpy().astype(np.float32)
                    keep = soft_nms(np_dets, cfg.TEST.SOFT_NMS_METHOD)  # np_dets will be changed in soft_nms
                    keep = torch.from_numpy(keep).type_as(cls_dets).int()
                    cls_dets = torch.from_numpy(np_dets).type_as(cls_dets)
                else:
                    keep = nms(cls_dets, cfg.TEST.NMS)
                cls_dets = cls_dets[keep.view(-1).long()]
                if vis:
                    # im2show = vis_detections(im2show, imdb.classes[j], cls_dets.cpu().numpy(), 0.3)
                    im2show = vis_detections(im2show, imdb.classes[j], cls_dets.cpu().numpy(), 0.0)
                    # im2show = vis_detections(im2show, imdb.classes[j], gt_boxes.data.cpu().numpy(), 0.0)
                all_boxes[j][i] = cls_dets.cpu().numpy()
            else:
                all_boxes[j][i] = empty_array

        # Limit to max_per_image detections *over all classes*
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1]
                                      for j in xrange(1, imdb.num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in xrange(1, imdb.num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]

        misc_toc = time.time()
        nms_time = misc_toc - misc_tic

        sys.stdout.write('[VAL]: im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
                         .format(i + 1, num_images, detect_time, nms_time))
        sys.stdout.flush()

        if vis:
            print(os.path.join(output_dir, 'result%d.png' % (i)))
            cv2.imwrite(os.path.join(output_dir, 'result%d.png' % (i)), im2show)

    fpn.train()
    return all_boxes

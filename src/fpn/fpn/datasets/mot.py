# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import configparser
import csv
import os
import os.path as osp
import pickle

# import PIL
import numpy as np
import scipy

from ..model.utils.config import cfg
from .imdb import imdb


class MOT17(imdb):
    """ Data class for the Multiple Object Tracking Dataset

    In the MOT dataset the different images are stored in folders belonging to the
    sequences. In every sequence the image index starts at 000001.jpg again. Therefore
    the image name can't be used directly as index and has to be mapped to a unique positive
    integer.

    Attributes:
        _image_index: List of mapped unique indexes of each image that will be used in this set

    """

    def __init__(self, image_set, year):
        imdb.__init__(self, 'mot_' + year + '_' + image_set)

        self._year = year
        self._image_set = image_set
        self._mot_dir = os.path.join(
            cfg.DATA_DIR, 'MOT' + self._year[2:4] + 'Det')
        self._mot_train_dir = os.path.join(self._mot_dir, 'train')
        self._mot_test_dir = os.path.join(self._mot_dir, 'test')
        self._roidb_handler = self.gt_roidb

        self._classes = ('__background__',  # always index 0
                         'pedestrian')
        #self._num_classes = len(self._classes)

        self._train_folders = ['MOT17-02', 'MOT17-04', 'MOT17-05', 'MOT17-09', 'MOT17-10', 'MOT17-11', 'MOT17-13']
        self._test_folders = ['MOT17-01', 'MOT17-03', 'MOT17-06', 'MOT17-07', 'MOT17-08', 'MOT17-12', 'MOT17-14']

        if 'seq' in self._image_set:
            split_num = int(self._image_set[-1])
            if 'train' in self._image_set:
                self._train_folders.pop(len(self._train_folders) - split_num)
            else:
                self._train_folders = [self._train_folders.pop(len(self._train_folders) - split_num)]

        assert os.path.exists(self._mot_dir), \
            'Path does not exist: {}'.format(self._mot_dir)
        assert os.path.exists(self._mot_train_dir), \
            'Path does not exist: {}'.format(self._mot_train_dir)
        assert os.path.exists(self._mot_test_dir), \
            'Path does not exist: {}'.format(self._mot_test_dir)

        self._index_to_path = {}
        self._index_to_width = {}
        self._index_to_height = {}

        counter = 0
        frame_train_set = []
        frame_val_set = []

        for f in self._train_folders:
            path = os.path.join(self._mot_train_dir, f)
            config_file = os.path.join(path, 'seqinfo.ini')

            assert os.path.exists(config_file), \
                'Path does not exist: {}'.format(config_file)

            config = configparser.ConfigParser()
            config.read(config_file)
            seqLength = int(config['Sequence']['seqLength'])
            imWidth = int(config['Sequence']['imWidth'])
            imHeight = int(config['Sequence']['imHeight'])
            imExt = config['Sequence']['imExt']
            imDir = config['Sequence']['imDir']

            _imDir = os.path.join(path, imDir)

            for i in range(1, seqLength+1):
                im_path = os.path.join(_imDir, ("{:06d}"+imExt).format(i))
                assert os.path.exists(im_path), \
                    'Path does not exist: {}'.format(im_path)
                self._index_to_path[i+counter] = im_path
                self._index_to_width[i+counter] = imWidth
                self._index_to_height[i+counter] = imHeight
                if i <= seqLength*0.5:
                    frame_train_set.append(i+counter)
                if i > seqLength*0.75:
                    frame_val_set.append(i+counter)

            counter += seqLength

        self._train_counter = counter

        for f in self._test_folders:
            path = os.path.join(self._mot_test_dir, f)
            config_file = os.path.join(path, 'seqinfo.ini')

            assert os.path.exists(config_file), \
                'Path does not exist: {}'.format(config_file)

            config = configparser.ConfigParser()
            config.read(config_file)
            seqLength = int(config['Sequence']['seqLength'])
            imWidth = int(config['Sequence']['imWidth'])
            imHeight = int(config['Sequence']['imHeight'])
            imExt = config['Sequence']['imExt']
            imDir = config['Sequence']['imDir']

            _imDir = os.path.join(path, imDir)

            for i in range(1, seqLength+1):
                im_path = os.path.join(_imDir, ("{:06d}"+imExt).format(i))
                assert os.path.exists(im_path), \
                    'Path does not exist: {}'.format(im_path)
                self._index_to_path[i+counter] = im_path
                self._index_to_width[i+counter] = imWidth
                self._index_to_height[i+counter] = imHeight

            counter += seqLength

        train_set = [i for i in range(1, self._train_counter+1)]
        test_set = [i for i in range(self._train_counter+1, counter+1)]
        all_set = [i for i in range(1, counter+1)]

        if self._image_set == "train" or "seq" in self._image_set:
            self._image_index = train_set
        elif self._image_set == "frame_train":
            self._image_index = frame_train_set
        elif self._image_set == "frame_val":
            self._image_index = frame_val_set
        elif self._image_set == "test":
            self._image_index = test_set
        elif self._image_set == "all":
            self._image_index = all_set

    def image_path_at(self, i):
        """
        Return the absolute path to image i (!= index, which begins at posivitve value and can
        have gaps) in the image sequence.
        0 based index
        """
        return self._index_to_path[self._image_index[i]]

    def image_id_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return i

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = pickle.load(fid)
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        gt_roidb = [self._load_mot_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb

    def _load_mot_annotation(self, index):
        """
        Loads the bounding boxes from the corresponding gt.txt files
        structure of a row in this file: <frame>, <id>, <bb left>, <bb top>, <bb width>,
        <bb height>, <conf>, <class>, <visibility>

        id: Id of the corresponding object (important for tracking not detection)
        conf: Confidence, only considering bb if this is 1
        class: 1 is class for pedestrian
        visibility: Only consider if this is > 0.25

        If image is from test set return empy elements
        """

        if index > self._train_counter:
            boxes = np.zeros((0, 4), dtype=np.uint16)
            gt_classes = np.zeros((0), dtype=np.int32)
            overlaps = np.zeros((0, self.num_classes), dtype=np.float32)
            overlaps = scipy.sparse.csr_matrix(overlaps)
            seg_areas = np.zeros((0), dtype=np.float32)
            visibilities = np.zeros((0), dtype=np.float32)

            return {'boxes': boxes,
                    'gt_classes': gt_classes,
                    #'gt_ishard': ishards,
                    'gt_overlaps': overlaps,
                    'flipped': False,
                    'seg_areas': seg_areas,
                    'visibilities': visibilities}

        image_pth = self._index_to_path[index]
        file_index = int(os.path.basename(image_pth).split('.')[0])

        gt_file = os.path.join(os.path.dirname(
            os.path.dirname(image_pth)), 'gt', 'gt.txt')

        assert os.path.exists(gt_file), \
            'Path does not exist: {}'.format(gt_file)

        bounding_boxes = []

        with open(gt_file, "r") as inf:
            reader = csv.reader(inf, delimiter=',')
            for row in reader:
                if int(row[0]) == file_index and int(row[6]) == 1 and int(row[7]) == 1 and float(row[8]) >= 0.25:
                    bb = {}
                    bb['bb_left'] = int(row[2])
                    bb['bb_top'] = int(row[3])
                    bb['bb_width'] = int(row[4])
                    bb['bb_height'] = int(row[5])
                    bb['visibility'] = float(row[8])

                    # Check if bb is inside the image
                    #if bb['bb_left'] > 0 and bb['bb_top'] > 0 \
                    #    and bb['bb_left']+bb['bb_width']-1 <= self._index_to_width[index] \
                    #    and bb['bb_top']+bb['bb_height']-1 <= self._index_to_height[index]:

                    # Now crop bb that are outside of image
                    #if bb['bb_left'] <= 0:
                    #    bb['bb_left'] = 1
                    #if bb['bb_top'] <= 0:
                    #    bb['bb_top'] = 1
                    #if bb['bb_left']+bb['bb_width']-1 > self._index_to_width[index]:
                    #    bb['bb_width'] = self._index_to_width[index] - bb['bb_left'] + 1
                    #if bb['bb_top']+bb['bb_height']-1 > self._index_to_height[index]:
                    #    bb['bb_height'] = self._index_to_height[index] - bb['bb_top'] + 1

                    bounding_boxes.append(bb)

        num_objs = len(bounding_boxes)

        #boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        boxes = np.zeros((num_objs, 4), dtype=np.int16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)
        #ishards = np.zeros((num_objs), dtype=np.int32)
        visibilities = np.zeros((num_objs), dtype=np.float32)

        for i, bb in enumerate(bounding_boxes):
            # Make pixel indexes 0-based, should already be 0-based (or not)
            x1 = bb['bb_left'] - 1
            y1 = bb['bb_top'] - 1
            # This -1 accounts for the width (width of 1 x1=x2)
            x2 = x1 + bb['bb_width'] - 1
            y2 = y1 + bb['bb_height'] - 1

            boxes[i, :] = [x1, y1, x2, y2]
            # Class is always pedestrian
            gt_classes[i] = 1
            seg_areas[i] = float((x2 - x1 + 1) * (y2 - y1 + 1))
            #ishards[i] = 0
            overlaps[i][1] = 1.0
            visibilities[i] = bb['visibility']

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes': boxes,
                'gt_classes': gt_classes,
                #'gt_ishard': ishards,
                'gt_overlaps': overlaps,
                'flipped': False,
                'seg_areas': seg_areas,
                'visibilities': visibilities}

    """
    def evaluate_detections(self, all_boxes, output_dir=None, ret=False):

        if self._image_set in ["test", "all"]:
            assert ret == False, 'Evaluating on test set, can\'t return values'
            assert output_dir, 'No output dir was given!'
            self._write_results_file(all_boxes, output_dir)
        else:
            tp, fp, prec, rec, ap = self._python_eval(all_boxes)
            results_string = "True Positives: {}\nFalse Positives: {}\nPrecision: {}\nRecall: {}\nAP: {}".format(tp, fp, prec, rec, ap)
            if output_dir:
                results_file = osp.join(output_dir, 'results_scores.txt')
                print("[*] Saving results to {}".format(results_file))
                with open(results_file, "w") as rf:
                    rf.write(results_string)
                self._write_results_file(all_boxes, output_dir)
            print(results_string)

        if ret:
            return ap, rec, prec
    """

    def evaluate_detections(self, all_boxes, output_dir):
        self._write_results_file(all_boxes, output_dir)
        tp, fp, prec, rec, ap = self._python_eval(all_boxes)
        print(f"AP: {ap} Prec: {prec} Rec: {rec} TP: {tp} FP: {fp}")
        return ap

    def _matlab_eval(self, all_boxes):
        pass

    def _python_eval(self, all_boxes, ovthresh=0.5):
        """Evaluates the detections (not official!!)

        all_boxes[cls][image] = N x 5 array of detections in (x1, y1, x2, y2, score)
        """

        gt_roidb = self.gt_roidb()

        # Lists for tp and fp in the format tp[cls][image]
        tp = [[[] for _ in range(len(self.image_index))]
              for _ in range(self.num_classes)]
        fp = [[[] for _ in range(len(self.image_index))]
              for _ in range(self.num_classes)]

        # sum up all positive groundtruth samples
        npos = 0

        for cls_index, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            det = all_boxes[cls_index]
            gt = []
            # keep track which groundtruth boxes have been detected already
            gt_found = []

            # Only keep gt data for right class
            for im_gt in gt_roidb:
                bbox = im_gt['boxes'][np.where(
                    np.logical_and(im_gt['gt_classes'] == cls_index,
                                   im_gt['visibilities'] >= 0.5))]
                found = np.zeros(bbox.shape[0])
                gt.append(bbox)
                gt_found.append(found)

                npos += found.shape[0]

            # Loop through all images
            for im_index, (im_det, im_gt, found) in enumerate(zip(det, gt, gt_found)):
                # Loop through dets an mark TPs and FPs

                im_tp = np.zeros(len(im_det))
                im_fp = np.zeros(len(im_det))
                for i, d in enumerate(im_det):
                    ovmax = -np.inf

                    if im_gt.size > 0:
                        # compute overlaps
                        # intersection
                        ixmin = np.maximum(im_gt[:, 0], d[0])
                        iymin = np.maximum(im_gt[:, 1], d[1])
                        ixmax = np.minimum(im_gt[:, 2], d[2])
                        iymax = np.minimum(im_gt[:, 3], d[3])
                        iw = np.maximum(ixmax - ixmin + 1., 0.)
                        ih = np.maximum(iymax - iymin + 1., 0.)
                        inters = iw * ih

                        # union
                        uni = ((d[2] - d[0] + 1.) * (d[3] - d[1] + 1.) +
                               (im_gt[:, 2] - im_gt[:, 0] + 1.) *
                               (im_gt[:, 3] - im_gt[:, 1] + 1.) - inters)

                        overlaps = inters / uni
                        ovmax = np.max(overlaps)
                        jmax = np.argmax(overlaps)

                    if ovmax > ovthresh:
                        if found[jmax] == 0:
                            im_tp[i] = 1.
                            found[jmax] = 1.
                        else:
                            im_fp[i] = 1.
                    else:
                        im_fp[i] = 1.

                tp[cls_index][im_index] = im_tp
                fp[cls_index][im_index] = im_fp

        # Flatten out tp and fp into a numpy array
        i = 0
        for cls in tp:
            for im in cls:
                if type(im) != type([]):
                    i += im.shape[0]

        tp_flat = np.zeros(i)
        fp_flat = np.zeros(i)

        i = 0
        for tp_cls, fp_cls in zip(tp, fp):
            for tp_im, fp_im in zip(tp_cls, fp_cls):
                if type(tp_im) != type([]):
                    s = tp_im.shape[0]
                    tp_flat[i:s+i] = tp_im
                    fp_flat[i:s+i] = fp_im
                    i += s

        tp = np.cumsum(tp_flat)
        fp = np.cumsum(fp_flat)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth (probably not needed in my code but doesn't harm if left)
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        tmp = np.maximum(tp + fp, np.finfo(np.float64).eps)

        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

        return np.max(tp), np.max(fp), prec[-1], np.max(rec), ap

    def _write_results_file(self, all_boxes, output_dir):
        """Write the detections in the format for MOT17Det sumbission

        all_boxes[cls][image] = N x 5 array of detections in (x1, y1, x2, y2, score)

        Each file contains these lines:
        <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>

        Files to sumbit:
        ./MOT17-01.txt
        ./MOT17-02.txt
        ./MOT17-03.txt
        ./MOT17-04.txt
        ./MOT17-05.txt
        ./MOT17-06.txt
        ./MOT17-07.txt
        ./MOT17-08.txt
        ./MOT17-09.txt
        ./MOT17-10.txt
        ./MOT17-11.txt
        ./MOT17-12.txt
        ./MOT17-13.txt
        ./MOT17-14.txt
        """

        #format_str = "{}, -1, {}, {}, {}, {}, {}, -1, -1, -1"

        files = {}
        for cls in all_boxes:
            for i, dets in enumerate(cls):
                path = self.image_path_at(i)
                img1, name = osp.split(path)
                # get image number out of name
                frame = int(name.split('.')[0])
                # smth like /train/MOT17-09-FRCNN or /train/MOT17-09
                tmp = osp.dirname(img1)
                # get the folder name of the sequence and split it
                tmp = osp.basename(tmp).split('-')
                # Now get the output name of the file
                out = tmp[0]+'-'+tmp[1]+'.txt'
                outfile = osp.join(output_dir, out)

                # check if out in keys and create empty list if not
                if outfile not in files.keys():
                    files[outfile] = []

                for d in dets:
                    x1 = d[0]
                    y1 = d[1]
                    x2 = d[2]
                    y2 = d[3]
                    score = d[4]
                    files[outfile].append(
                        [frame, -1, x1+1, y1+1, x2-x1+1, y2-y1+1, score, -1, -1, -1])

        for k, v in files.items():
            #outfile = osp.join(output_dir, out)
            with open(k, "w") as of:
                writer = csv.writer(of, delimiter=',')
                for d in v:
                    writer.writerow(d)


class MOT19CVPR(MOT17):

    def __init__(self, image_set):
        imdb.__init__(self, f'mot19_{image_set}')

        self._image_set = image_set
        self._mot_dir = os.path.join(
            cfg.DATA_DIR, 'MOT19_CVPR')
        self._mot_train_dir = os.path.join(self._mot_dir, 'train')
        self._mot_test_dir = os.path.join(self._mot_dir, 'test')
        self._roidb_handler = self.gt_roidb

        self._classes = ('__background__',  # always index 0
                         'pedestrian')
        #self._num_classes = len(self._classes)

        self._train_folders = ['CVPR19-01', 'CVPR19-02', 'CVPR19-03', 'CVPR19-05']
        self._test_folders = ['CVPR19-04', 'CVPR19-06', 'CVPR19-07', 'CVPR19-08']

        if 'seq' in self._image_set:
            split_num = int(self._image_set[-1])
            if 'train' in self._image_set:
                self._train_folders.pop(len(self._train_folders) - split_num)
            else:
                self._train_folders = [self._train_folders.pop(
                    len(self._train_folders) - split_num)]

        assert os.path.exists(self._mot_dir), \
            'Path does not exist: {}'.format(self._mot_dir)
        assert os.path.exists(self._mot_train_dir), \
            'Path does not exist: {}'.format(self._mot_train_dir)
        assert os.path.exists(self._mot_test_dir), \
            'Path does not exist: {}'.format(self._mot_test_dir)

        self._index_to_path = {}
        self._index_to_width = {}
        self._index_to_height = {}

        counter = 0
        frame_train_set = []
        frame_val_set = []

        for f in self._train_folders:
            path = os.path.join(self._mot_train_dir, f)
            config_file = os.path.join(path, 'seqinfo.ini')

            assert os.path.exists(config_file), \
                'Path does not exist: {}'.format(config_file)

            config = configparser.ConfigParser()
            config.read(config_file)
            seqLength = int(config['Sequence']['seqLength'])
            imWidth = int(config['Sequence']['imWidth'])
            imHeight = int(config['Sequence']['imHeight'])
            imExt = config['Sequence']['imExt']
            imDir = config['Sequence']['imDir']

            _imDir = os.path.join(path, imDir)

            for i in range(1, seqLength + 1):
                im_path = os.path.join(_imDir, ("{:06d}" + imExt).format(i))
                assert os.path.exists(im_path), \
                    'Path does not exist: {}'.format(im_path)
                self._index_to_path[i + counter] = im_path
                self._index_to_width[i + counter] = imWidth
                self._index_to_height[i + counter] = imHeight
                if i <= seqLength * 0.5:
                    frame_train_set.append(i + counter)
                if i > seqLength * 0.75:
                    frame_val_set.append(i + counter)

            counter += seqLength

        self._train_counter = counter

        for f in self._test_folders:
            path = os.path.join(self._mot_test_dir, f)
            config_file = os.path.join(path, 'seqinfo.ini')

            assert os.path.exists(config_file), \
                'Path does not exist: {}'.format(config_file)

            config = configparser.ConfigParser()
            config.read(config_file)
            seqLength = int(config['Sequence']['seqLength'])
            imWidth = int(config['Sequence']['imWidth'])
            imHeight = int(config['Sequence']['imHeight'])
            imExt = config['Sequence']['imExt']
            imDir = config['Sequence']['imDir']

            _imDir = os.path.join(path, imDir)

            for i in range(1, seqLength + 1):
                im_path = os.path.join(_imDir, ("{:06d}" + imExt).format(i))
                assert os.path.exists(im_path), \
                    'Path does not exist: {}'.format(im_path)
                self._index_to_path[i + counter] = im_path
                self._index_to_width[i + counter] = imWidth
                self._index_to_height[i + counter] = imHeight

            counter += seqLength

        train_set = [i for i in range(1, self._train_counter + 1)]
        test_set = [i for i in range(self._train_counter + 1, counter + 1)]
        all_set = [i for i in range(1, counter + 1)]

        if self._image_set == "train" or "seq" in self._image_set:
            self._image_index = train_set
        elif self._image_set == "frame_train":
            self._image_index = frame_train_set
        elif self._image_set == "frame_val":
            self._image_index = frame_val_set
        elif self._image_set == "test":
            self._image_index = test_set
        elif self._image_set == "all":
            self._image_index = all_set


if __name__ == '__main__':
    pass

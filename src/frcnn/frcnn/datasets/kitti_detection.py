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

import numpy as np
import scipy
from PIL import Image

from ..model.config import cfg
from .imdb import imdb


class kitti_detection(imdb):
    """ Data class for the KITTI Tracking dataset

    """


    def __init__(self, image_set, cl):
        imdb.__init__(self, 'kitti_detection_{}_{}'.format(cl, image_set))

        self._cl = cl
        self._image_set = image_set
        self._kitti_dir = os.path.join(cfg.DATA_DIR, 'KITTI_detection')
        self._kitti_train_dir = os.path.join(self._kitti_dir, 'training')
        self._kitti_test_dir = os.path.join(self._kitti_dir, 'testing')
        self._roidb_handler = self.gt_roidb

        self._classes = ('__background__',  # always index 0
                         cl)
        #self._num_classes = len(self._classes)

        assert os.path.exists(self._kitti_dir), \
            'Path does not exist: {}'.format(self._kitti_dir)
        assert os.path.exists(self._kitti_train_dir), \
            'Path does not exist: {}'.format(self._kitti_train_dir)
        assert os.path.exists(self._kitti_test_dir), \
            'Path does not exist: {}'.format(self._kitti_test_dir)

        self._index_to_path = {}
        self._index_to_width = {}
        self._index_to_height = {}

        small_train_set = []
        small_val_set = []


        # list all training images
        train_image_dir = os.path.join(self._kitti_train_dir, 'image_2')
        #train_images = [f for f in os.listdir(train_image_dir) if "png" in f]
        train_images = ["%06d.png"%n for n in range(7481)]

        for i, im_name in enumerate(train_images, 1):
            im_path = os.path.join(train_image_dir, im_name)
            assert os.path.exists(im_path), \
                'Path does not exist: {}'.format(im_path)
            im = Image.open(im_path)
            width, height = im.size

            self._index_to_path[i] = im_path
            self._index_to_width[i] = width
            self._index_to_height[i] = height

            if i%5:
                small_train_set.append(i)
            else:
                small_val_set.append(i)

            self._train_counter = i

        # list all test images
        test_image_dir = os.path.join(self._kitti_test_dir, 'image_2')
        #test_images = [f for f in os.listdir(test_image_dir) if "png" in f]
        test_images = ["%06d.png"%n for n in range(7518)]

        test_counter = 0
        for i, im_name in enumerate(test_images, self._train_counter+1):
            im_path = os.path.join(test_image_dir, im_name)
            assert os.path.exists(im_path), \
                'Path does not exist: {}'.format(im_path)
            im = Image.open(im_path)
            width, height = im.size

            self._index_to_path[i] = im_path
            self._index_to_width[i] = width
            self._index_to_height[i] = height

            test_counter = i

        train_set = [i for i in range(1, self._train_counter+1)]
        test_set = [i for i in range(self._train_counter+1, self._train_counter+test_counter+1)]

        if self._image_set == "train":
            self._image_index = train_set
        elif self._image_set == "small_val":
            self._image_index = small_val_set
        elif self._image_set == "small_train":
            self._image_index = small_train_set
        elif self._image_set == "test":
            self._image_index = test_set

    def image_path_at(self, i):
        """
        Return the absolute path to image i (!= index, which begins at posivitve value and can
        have gaps) in the image sequence.
        0 based index
        """
        return self._index_to_path[self._image_index[i]]


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

        gt_roidb = [self._load_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb

    def _load_annotation(self, index):
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

            return {'boxes': boxes,
                'gt_classes': gt_classes,
                #'gt_ishard': ishards,
                'gt_overlaps': overlaps,
                'flipped': False,
                'seg_areas': seg_areas}


        image_pth = self._index_to_path[index]
        file_index = int(os.path.basename(image_pth).split('.')[0])

        gt_file = os.path.join(self._kitti_train_dir, 'label_2', '%06d.txt'%file_index)

        assert os.path.exists(gt_file), \
                'Path does not exist: {}'.format(gt_file)

        bounding_boxes = []

        with open(gt_file, "r") as inf:
            reader = csv.reader(inf, delimiter=' ')
            for row in reader:
                if row[0] == self._cl and float(row[1]) <= 0.5 and int(row[2]) <= 1:
                    bb = {}
                    bb['bb_left'] = int(float(row[4]))
                    bb['bb_top'] = int(float(row[5]))
                    bb['bb_right'] = int(float(row[6]))
                    bb['bb_bottom'] = int(float(row[7]))

                    bounding_boxes.append(bb)

        num_objs = len(bounding_boxes)

        #boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        boxes = np.zeros((num_objs, 4), dtype=np.int16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)
        #ishards = np.zeros((num_objs), dtype=np.int32)

        for i, bb in enumerate(bounding_boxes):
            # Make pixel indexes 0-based, should already be 0-based
            #x1 = bb['bb_left'] - 1
            #y1 = bb['bb_top'] - 1
            # This -1 accounts for the width (width of 1 x1=x2)
            #x2 = x1 + bb['bb_width'] - 1
            #y2 = y1 + bb['bb_height'] - 1

            x1 = bb['bb_left']
            y1 = bb['bb_top']
            x2 = bb['bb_right']
            y2 = bb['bb_bottom']

            boxes[i, :] = [x1, y1, x2, y2]
            # Class is always pedestrian
            gt_classes[i] = 1
            seg_areas[i] = float((x2 - x1 + 1) * (y2 - y1 + 1))
            #ishards[i] = 0
            overlaps[i][1] = 1.0


        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes': boxes,
                'gt_classes': gt_classes,
                #'gt_ishard': ishards,
                'gt_overlaps': overlaps,
                'flipped': False,
                'seg_areas': seg_areas}

    def evaluate_detections(self, all_boxes, output_dir):

        self._write_results_file(all_boxes, output_dir)

    def _matlab_eval(self, all_boxes):
        pass

    def _python_eval(self, all_boxes, ovthresh=0.5):
        pass

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

        # assure everything is put out in order to evaluate small splits
        if "test" not in self._image_set:
            for i in range(7481):
                out = '%06d.txt'%(i)
                outfile = osp.join(output_dir, out)
                files[outfile] = []

        for cls in all_boxes:
            for i, dets in enumerate(cls):
                path = self.image_path_at(i)
                _, name = osp.split(path)
                # get image number out of name
                frame = int(name.split('.')[0])
                # Now get the output name of the file
                out = '%06d.txt'%(frame)
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
                    files[outfile].append([self._cl, 0, 0, 0, x1, y1, x2, y2, 0, 0, 0,
                        0, 0, 0, 0, score])

        for k,v in files.items():
            #outfile = osp.join(output_dir, out)
            with open(k, "w") as of:
                writer = csv.writer(of, delimiter=' ')
                for d in v:
                    writer.writerow(d)


if __name__ == '__main__':
    pass

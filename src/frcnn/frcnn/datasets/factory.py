# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""
from __future__ import absolute_import, division, print_function

import numpy as np

# from .coco import coco
from .kitti_detection import kitti_detection
from .mot import mot
from .pascal_voc import pascal_voc

__sets = {}


# Set up voc_<year>_<split>
for year in ['2007', '2012']:
  for split in ['train', 'val', 'trainval', 'test']:
    name = 'voc_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))

for year in ['2007', '2012']:
  for split in ['train', 'val', 'trainval', 'test']:
    name = 'voc_{}_{}_diff'.format(year, split)
    __sets[name] = (lambda split=split, year=year: pascal_voc(split, year, use_diff=True))

# Set up coco_2014_<split>
# for year in ['2014']:
#   for split in ['train', 'val', 'minival', 'valminusminival', 'trainval']:
#     name = 'coco_{}_{}'.format(year, split)
#     __sets[name] = (lambda split=split, year=year: coco(split, year))

# # Set up coco_2015_<split>
# for year in ['2015']:
#   for split in ['test', 'test-dev']:
#     name = 'coco_{}_{}'.format(year, split)
#     __sets[name] = (lambda split=split, year=year: coco(split, year))

# MOT17 dataset
for year in ['2017', '2019']:
  for split in ['train', 'small_val', 'small_train', 'test', 'all']:
    name = 'mot_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: mot(split, year))

# KITTI object
for cl in ['Car', 'Pedestrian', 'Cyclist']:
  for split in ['train', 'small_val', 'small_train', 'test']:
    name = 'kitti_detection_{}_{}'.format(cl, split)
    __sets[name] = (lambda split=split, cl=cl: kitti_detection(split, cl))

def get_imdb(name):
  """Get an imdb (image database) by name."""
  if name not in __sets:
    raise KeyError('Unknown dataset: {}'.format(name))
  return __sets[name]()


def list_imdbs():
  """List all registered imdbs."""
  return list(__sets.keys())

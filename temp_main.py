from __future__ import absolute_import, division, print_function
import cv2 
import os
import os.path as osp
import pprint
import time

import numpy as np
import torch
import yaml
from torch.autograd import Variable
from torch.utils.data import DataLoader

from sacred import Experiment
from tracktor.config import get_output_dir
from tracktor.resnet import resnet50
from tracktor.tracker import Tracker
from frcnn.model import test
from tracktor.utils import interpolate, plot_sequence

import data_handle
ex = Experiment()

ex.add_config('experiments/cfgs/tracktor.yaml')
ex.add_config(ex.configurations[0]._conf['tracktor']['obj_detect_config'])
ex.add_config(ex.configurations[0]._conf['tracktor']['reid_network_config'])

webcam = 'data/boli_pianduan.mp4'

@ex.automain
def main(tracktor,siamese, _config):
    # set all seeds
    
    torch.manual_seed(tracktor['seed'])
    torch.cuda.manual_seed(tracktor['seed'])
    np.random.seed(tracktor['seed'])
    torch.backends.cudnn.deterministic = True

    output_dir = osp.join(get_output_dir(tracktor['module_name']), tracktor['name'])
    sacred_config = osp.join(output_dir, 'sacred_config.yaml')
    if not osp.exists(output_dir):
        os.makedirs(output_dir)
    with open(sacred_config, 'w') as outfile:
        yaml.dump(_config, outfile, default_flow_style=False)
    ##########################
    # Initialize the modules #
    ##########################

    # object detection
    print("[*] Building object detector")
    if tracktor['network'].startswith('frcnn'):
        # FRCNN
        from tracktor.frcnn import FRCNN
        from frcnn.model import config

        if _config['frcnn']['cfg_file']:
            config.cfg_from_file(_config['frcnn']['cfg_file'])
        if _config['frcnn']['set_cfgs']:
            config.cfg_from_list(_config['frcnn']['set_cfgs'])

        obj_detect = FRCNN(num_layers=101)
        obj_detect.create_architecture(2, tag='default',
            anchor_scales=config.cfg.ANCHOR_SCALES,
            anchor_ratios=config.cfg.ANCHOR_RATIOS)
        obj_detect.load_state_dict(torch.load(tracktor['obj_detect_weights']))
    else:
        raise NotImplementedError(f"Object detector type not known: {tracktor['network']}")
    obj_detect.eval()
    obj_detect.cuda()

    # tracktor
    tracker = Tracker(obj_detect, tracktor['tracker'])
    tracker.reset()  # init tracker

    print("[*] Beginning evaluation...")
    time_total = 0
    cap = cv2.VideoCapture(webcam)
    num_images = 0
    images = []
    try:
        begin = time.time()
        while (cap.isOpened()):
            ret, frame = cap.read()
            images.append(frame)
            time.time()
            try:
                blob = data_handle.data_process(frame)
            except:
                print('over')
                break
            tracker.step(blob)
            num_images += 1
            if num_images % 10 == 0:
                print('now is :',num_images)
        results = tracker.get_results()
        end = time.time()
        print("[*] Tracks found: {}".format(len(results)))
        print('It takes: {:.3f} s'.format((end-begin)))
        if tracktor['write_images']:
            plot_sequence(results, images, '/home/longshuz/project/tracking_wo_bnw/output/tracktor/results')
        cap.release()

    except:
        raise KeyboardInterrupt
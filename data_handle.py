from frcnn.model import test
import torch
import numpy as np
from torchvision.transforms import Compose, Normalize, ToTensor
from PIL import Image
import cv2
import copy

normalize_mean=[0.485, 0.456, 0.406]
normalize_std=[0.229, 0.224, 0.225]
transforms = Compose([ToTensor(), Normalize(normalize_mean,normalize_std)])

def data_process(frame):
    im = np.array(frame)
    
    data, im_scales = test._get_image_blob(frame)
    c = np.array([data.shape[1], data.shape[2], im_scales[0]],dtype=np.float32)
    c = torch.from_numpy(c).float()
    data = torch.from_numpy(data)

    sample = {}
    sample['data'] = torch.unsqueeze(data,0)
    sample['im_info'] = torch.unsqueeze(c,0)
    # convert to siamese input
    im = Image.fromarray(im)
    im = transforms(im).unsqueeze(0)
    sample['app_data'] = im.unsqueeze(0)
    return sample                                 
# extract text or image feature
import os

import torch
import numpy as np
import argparse
import clip
from PIL import Image
from datasets.dataset import CustomDataset
import datasets.transforms as T
from torchvision import datasets, transforms
import torchvision.models.resnet as models
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.misc import collate_fn_new
import pickle


def save_pkl(obj, p):
    with open(p, "wb") as f:
        pickle.dump(obj, f)


def load_pkl(p):
    with open(p, "rb") as f:
        res = pickle.load(f)
    return res


def filter_bbox_by_area(image, boxes, bbox_area_threshold=0.001):
    w, h = image.size
    image_area = w * h
    new_bbox = []
    for i in range(len(boxes)):
        bbox = boxes[i]
        bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        area_ratio = bbox_area / image_area
        if area_ratio < bbox_area_threshold:
            continue
        else:
            new_bbox.append(bbox)
    return new_bbox


if __name__ == '__main__':
    datatypes = ['CUB', 'AWA2', 'SUN']
    transform = T.Compose([
        T.RandomResize([448]),
        T.CenterCrop((448, 448)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    for datatype in datatypes:
        imagename2bbox_path = f'../data/{datatype}/imagename2bbox.pkl'
        imagename2bbox = load_pkl(imagename2bbox_path)
        new_imagename2bbox = {}
        for imagename, param_bbox in imagename2bbox.items():
            img = Image.open(imagename).convert('RGB')
            search_dict = {}
            for param_set, boxes in param_bbox.items():
                temp_boxes = boxes.copy()
                temp_boxes = filter_bbox_by_area(img, temp_boxes)
                temp_boxes = torch.Tensor(temp_boxes)
                _, target = transform(img, {'boxes': temp_boxes})
                search_dict[param_set] = target['boxes'].numpy()
            new_imagename2bbox[imagename] = search_dict
        imagename2bbox_save_path = f'../data/{datatype}/pretrained_imagename2bbox.pkl'
        save_pkl(new_imagename2bbox, imagename2bbox_save_path)

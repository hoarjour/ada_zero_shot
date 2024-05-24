import numpy as np
import random
import torch
from torch.utils.data import Dataset
from scipy import io
from sklearn import preprocessing
import os
import pickle
import cv2
from PIL import Image
import datasets.transforms as T


def load_pkl(p):
    with open(p, "rb") as f:
        res = pickle.load(f)
    return res


class CUBDataset(Dataset):

    def __init__(self, feature_path, metadata_path, is_train=True, class_feature_type='w2v'):
        super().__init__()
        self.feature_path = feature_path
        metadata = load_pkl(metadata_path)
        self.train_label = metadata['train_label']
        self.test_label = metadata['test_label']
        self.clip_class_features = metadata['clip_class_features']
        self.w2v_class_features = metadata['w2v_class_features']
        self.is_train = is_train
        self.class_feature_type = class_feature_type

    def __getitem__(self, idx):
        if self.is_train:
            feature_path = os.path.join(self.feature_path, 'train', f'{idx}_train.npy')
            label = self.train_label[idx]
        else:
            feature_path = os.path.join(self.feature_path, 'test', f'{idx}_test.npy')
            label = self.test_label[idx]
        feature = np.load(feature_path)
        feature = torch.from_numpy(feature)

        if self.class_feature_type == 'w2v':
            class_feature = self.w2v_class_features[label]
        elif self.class_feature_type == 'clip':
            class_feature = self.clip_class_features[label]
        else:
            raise NotImplementedError(f"不支持的class_feature_type {self.class_feature_type}")
        class_feature = torch.from_numpy(class_feature)

        return feature, label, class_feature

    def __len__(self):
        if self.is_train:
            return self.train_label.__len__()
        else:
            return self.test_label.__len__()


class CustomDataset(Dataset):
    def __init__(self, metadata_path, imagename2bbox_path, transform=None, markers=9, compactness=0.001,
                 bbox_area_threshold=0.1,
                 class_feature_type='w2v',
                 is_train=True):
        metadata = load_pkl(metadata_path)
        self.imagename2bbox = load_pkl(imagename2bbox_path)
        self.transform = transform
        self.markers = markers
        self.compactness = compactness

        self.bbox_area_threshold = bbox_area_threshold
        self.is_train = is_train
        assert class_feature_type in ['clip', 'w2v']
        self.class_feature_type = class_feature_type

        self.class_to_clip_feature = metadata['class_to_clip_feature']
        self.class_to_w2v_feature = metadata['class_to_w2v_feature']

        self.class_to_original_att = metadata['class_to_original_att']

        self.train_labels = metadata['train_labels']
        self.test_unseen_labels = metadata['test_unseen_labels']
        self.test_seen_labels = metadata['test_seen_labels']

        self.train_classes = metadata['train_classes']
        self.test_unseen_classes = metadata['test_unseen_classes']
        self.test_seen_classes = metadata['test_seen_classes']

        self.train_paths = metadata['train_paths']
        self.test_unseen_paths = metadata['test_unseen_paths']
        self.test_seen_paths = metadata['test_seen_paths']

        # for k, v in self.metadata.items():
        #     setattr(self, k, v)

    def filter_bbox_by_area(self, image, boxes):
        w, h = image.size
        image_area = w * h
        new_bbox = []
        for i in range(len(boxes)):
            bbox = boxes[i]
            bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            area_ratio = bbox_area / image_area
            if area_ratio < self.bbox_area_threshold:
                continue
            else:
                new_bbox.append(bbox)
        return new_bbox

    def __len__(self):
        if self.is_train:
            return self.train_paths.__len__()
        else:
            return self.test_seen_paths.__len__()

    def __getitem__(self, idx):
        if self.is_train:
            image_path = self.train_paths[idx]
            label = self.train_labels[idx]
            classname = self.train_classes[idx]
        else:
            image_path = self.test_seen_paths[idx]
            label = self.test_seen_labels[idx]
            classname = self.test_seen_classes[idx]
        classname = classname[0]
        if self.class_feature_type == 'w2v':
            class_feature = self.class_to_w2v_feature[classname]
            class_feature = torch.from_numpy(class_feature).float()
        else:
            class_feature = self.class_to_clip_feature[classname]

        original_attr = self.class_to_original_att[classname]
        original_attr = torch.from_numpy(original_attr)

        image = Image.open(image_path).convert("RGB")

        w, h = image.size

        bbox_key1 = "../" + image_path
        all_boxes = self.imagename2bbox[bbox_key1]  # xyxy

        bbox_key2 = f'markers:{self.markers},compactness:{self.compactness}'
        if bbox_key2 not in all_boxes:
            raise KeyError(f"没有这个参数组合: markers:{self.markers}, compactness:{self.compactness}")
        boxes = all_boxes[bbox_key2]

        boxes = self.filter_bbox_by_area(image, boxes)

        target = {
            'label': label[0],
            'class_feature': class_feature,
            'boxes': torch.Tensor(boxes),
            'original_attr': original_attr,
            'original_size': torch.Tensor([h, w])
        }

        if self.transform is not None:
            image, target = self.transform(image, target)

        return image, target


def make_coco_transforms(is_train):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if is_train:
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])
    else:
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])


def build_dataset(args, is_train):
    metadata_path = f'./data/{args.dataset}/meta_data.pkl'
    imagename2bbox_path = f'./data/{args.dataset}/imagename2bbox.pkl'
    dataset = CustomDataset(metadata_path=metadata_path,
                            imagename2bbox_path=imagename2bbox_path,
                            transform=make_coco_transforms(is_train),
                            markers=args.markers,
                            compactness=args.compactness,
                            bbox_area_threshold=args.bbox_area_threshold,
                            class_feature_type=args.class_feature_type,
                            is_train=is_train
                            )

    return dataset

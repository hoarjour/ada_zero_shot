import numpy as np
import random
import torch
from torch.utils.data import Dataset
from scipy import io
from sklearn import preprocessing
import os
import pickle


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


def build_dataset(args, is_train):
    dataset = CUBDataset(feature_path=args.feature_path,
                         metadata_path=args.metadata_path,
                         is_train=is_train,
                         class_feature_type=args.class_feature_type)
    return dataset

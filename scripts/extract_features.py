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

class TempDataset(CustomDataset):
    def __init__(self, metadata_path, imagename2bbox_path, transform, image_split_set=None):
        super().__init__(metadata_path, imagename2bbox_path, transform)
        self.paths = getattr(self, f'{image_split_set}_paths')
        self.labels = getattr(self, f'{image_split_set}_labels')
        self.classes = getattr(self, f'{image_split_set}_classes')

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image_path = self.paths[idx]
        bbox_key1 = "../" + image_path

        image = Image.open(bbox_key1).convert("RGB")
        all_boxes = self.imagename2bbox[bbox_key1]  # xyxy
        bbox_key2 = f'markers:{self.markers},compactness:{self.compactness}'
        if bbox_key2 not in all_boxes:
            raise KeyError(f"没有这个参数组合: markers:{self.markers}, compactness:{self.compactness}")
        boxes = all_boxes[bbox_key2]

        boxes = self.filter_bbox_by_area(image, boxes)
        target = {
            'boxes': torch.Tensor(boxes),
            'path': image_path
        }
        if self.transform is not None:
            image, target = self.transform(image, target)
        return image, target


if __name__ == '__main__':
    datatypes = ['CUB', 'AWA2', 'SUN']
    image_split_sets = ['train', 'test_seen', 'test_unseen']
    transform = T.Compose([
        T.RandomResize([448]),
        T.CenterCrop((448, 448)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # region feature extractor
    device = torch.device('cuda')
    resnet101 = models.resnet101(pretrained=True).to(device)
    resnet101 = nn.Sequential(*list(resnet101.children())[:-2]).eval()

    for datatype in datatypes:
        for image_split_set in image_split_sets:
            metadata_path = f'../data/{datatype}/meta_data.pkl'
            imagename2bbox_path = f'../data/{datatype}/imagename2bbox.pkl'

            temp_dataset = TempDataset(metadata_path, imagename2bbox_path, transform, image_split_set)
            dataset_loader = torch.utils.data.DataLoader(temp_dataset, batch_size=32, collate_fn=collate_fn_new,
                                                         shuffle=False, num_workers=2)

            # all_features = {}
            temp_save_dir = f'../data/{datatype}/pretrained_features'
            os.makedirs(temp_save_dir, exist_ok=True)
            with torch.no_grad():
                for idx, (image, targets) in enumerate(dataset_loader):
                    image = torch.stack(image)
                    image = image.to(device)
                    features = resnet101(image).cpu().numpy()
                    paths = [t['path'] for t in targets]
                    for i in range(len(paths)):
                        path = paths[i]
                        filename = path.split("/")[-1].replace(".jpg", ".npy")
                        feature = features[i]

                        temp_save_path = os.path.join(temp_save_dir, filename)
                        np.save(temp_save_path, feature)

                    # feature_dict = dict(zip(paths, features))
                    # all_features.update(feature_dict)

            # temp_save_path = os.path.join(temp_save_dir, f'{image_split_set}.pkl')
            # save_pkl(all_features, temp_save_path)
            print(f"数据集 {datatype} 数据类型 {image_split_set} 提取特征成功，保存在 {temp_save_dir}")


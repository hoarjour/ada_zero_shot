import pickle
import scipy
import os
import numpy as np
import clip
import torch

prompt_dict = {
    "CUB": "A type of bird called [CLASS].",
    "AWA2": "An image showing [CLASS] in its natural habitat.",
    "SUN": "A picture showing [CLASS]."
}


def load_pkl(p):
    with open(p, "rb") as f:
        res = pickle.load(f)
    return res


def save_pkl(obj, p):
    with open(p, "wb") as f:
        pickle.dump(obj, f)


def prepare_metadata(dataset_type):
    meta_data = {}  # train_image_path_list, val_image_path_list, w2v, clip, train_label, val_label

    w2v_feature_path = f"../data/{dataset_type}/word2vec_splits.mat"
    w2v_attr = scipy.io.loadmat(w2v_feature_path)

    # clip feature
    clip_model, _ = clip.load('ViT-B/16')
    device = torch.device("cuda")
    clip_model = clip_model.to(device)

    all_classnames = w2v_attr['allclasses_names']
    all_classnames = np.squeeze(all_classnames)

    if dataset_type == 'CUB':
        all_classnames = [x[0].split('.')[-1] for x in all_classnames]
    else:
        all_classnames = [x[0] for x in all_classnames]
    # 增加prompt
    prompt = prompt_dict[dataset_type]

    prompted_classes = [prompt.replace("[CLASS]", x) for x in all_classnames]

    text_token = clip.tokenize(prompted_classes).to(device)
    with torch.no_grad():
        clip_text_features = clip_model.encode_text(text_token).float().cpu()
    # clip_text_features /= clip_text_features.norm(dim=-1, keepdim=True)  # 不进行norm
    meta_data['class_to_clip_feature'] = dict(zip(all_classnames, torch.unbind(clip_text_features, 0)))

    # w2v feature
    w2v_feature = w2v_attr['att'].T
    meta_data['class_to_w2v_feature'] = dict(zip(all_classnames, w2v_feature))

    # 图片的train, val split以及label
    feature_file_path = f'../data/{dataset_type}/res101.mat'
    attr_split_path = f'../data/{dataset_type}/att_splits.mat'

    feature_dict = scipy.io.loadmat(feature_file_path)
    attr_split = scipy.io.loadmat(attr_split_path)

    # original attribute
    ori_attr = attr_split['att'].T
    meta_data['class_to_original_att'] = dict(zip(all_classnames, ori_attr))

    all_image_names = feature_dict['image_files']
    raw_image_paths = [x[0][0] for x in all_image_names]
    if dataset_type == 'CUB':
        sub_image_paths = ['/'.join(x.split('/')[-4:]) for x in raw_image_paths]
        real_image_paths = [f'data/CUB/{x}' for x in sub_image_paths]
    elif dataset_type == 'AWA2':
        sub_image_paths = ['/'.join(x.split('/')[-4:]) for x in raw_image_paths]
        real_image_paths = [f'data/AWA2/AWA2-data/{x}' for x in sub_image_paths]
    elif dataset_type == 'SUN':
        real_image_paths = [x.replace('/BS/Deep_Fragments/work/MSc/data/SUN', 'data/SUN/SUNAttributeDB_Images') for x in raw_image_paths]
    else:
        raise Exception

    all_labels = feature_dict['labels']
    train_labels = all_labels[np.squeeze(attr_split['trainval_loc']-1)]
    test_unseen_labels = all_labels[np.squeeze(attr_split['test_unseen_loc']-1)]
    test_seen_labels = all_labels[np.squeeze(attr_split['test_seen_loc']-1)]

    train_classes = np.array(all_classnames)[train_labels - 1]
    test_unseen_classes = np.array(all_classnames)[test_unseen_labels - 1]
    test_seen_classes = np.array(all_classnames)[test_seen_labels - 1]

    real_image_paths = np.array(real_image_paths)
    train_paths = real_image_paths[np.squeeze(attr_split['trainval_loc'] - 1)]  # 185
    test_unseen_paths = real_image_paths[np.squeeze(attr_split['test_unseen_loc'] - 1)]  # 4
    test_seen_paths = real_image_paths[np.squeeze(attr_split['test_seen_loc'] - 1)]  # 153

    train_labels_seen = np.unique(train_labels)

    # new train label
    k = 0
    for label in train_labels_seen:
        train_labels[train_labels == label] = k
        k = k + 1

    meta_data['train_labels'] = train_labels
    meta_data['test_unseen_labels'] = test_unseen_labels
    meta_data['test_seen_labels'] = test_seen_labels

    meta_data['train_classes'] = train_classes
    meta_data['test_unseen_classes'] = test_unseen_classes
    meta_data['test_seen_classes'] = test_seen_classes

    meta_data['train_paths'] = train_paths
    meta_data['test_unseen_paths'] = test_unseen_paths
    meta_data['test_seen_paths'] = test_seen_paths

    save_path = f'../data/{dataset_type}/meta_data.pkl'
    save_pkl(meta_data, save_path)


if __name__ == '__main__':
    for data_type in ['CUB', 'AWA2', 'SUN']:
        prepare_metadata(data_type)

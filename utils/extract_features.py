# extract text or image feature
import torch
import numpy as np
import argparse
import clip


def clip_features(model, data_loader, source_type='text'):
    model.eval()
    pass


def get_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_type', type=str, default='text')
    parser.add_argument('--model_type', type=str, default='clip')
    parser.add_argument('--out_dir', type=str, default='./output')

    return parser.parse_args()

if __name__ == '__main__':
    pass


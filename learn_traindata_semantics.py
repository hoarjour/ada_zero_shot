import argparse
import torch
import numpy as np
import random
import os

from torch.utils.data import DataLoader
from datetime import datetime, timedelta

from datasets import build_dataset
from models import build_semantics_model, build_criterion
from utils.misc import collate_fn_new, collate_fn

parser = argparse.ArgumentParser(description="ZSL")

parser.add_argument('--dataset', help='choose between APY, AWA2, AWA1, CUB, SUN', default='CUB', type=str)
parser.add_argument('--epochs', default=300, type=int)
parser.add_argument('--lr', default=0.01, type=float)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--rand_seed', default=42, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--num_workers', default=1, type=int)
parser.add_argument('--norm_type', help='std(standard), L2, None', default='std', type=str)
parser.add_argument('--desc', default="default", type=str)

# Path
parser.add_argument("--feature_path", type=str, default="./data/CUB/resnet_features")
parser.add_argument("--metadata_path", type=str, default="./data/CUB/meta_data.pkl")
parser.add_argument("--imagename2bbox_path", type=str, default="./data/CUB/imagename2bbox.pkl")


parser.add_argument("--class_feature_type", type=str, default="w2v", choices=['clip', 'w2v'])
parser.add_argument('--markers', default=9, type=int)
parser.add_argument('--compactness', default=0.001, type=float)
parser.add_argument('--bbox_area_threshold', default=0.01, type=float,
                    help='the bbox area ratio threshold compared to whole image area')

# Loss coef
parser.add_argument('--class_loss_coef', default=1, type=float)
parser.add_argument('--relatedness_loss_coef', default=0, type=float)
parser.add_argument('--penalty_coef', default=0, type=float)

# model args
parser.add_argument('--num_queries', default=100, type=int)
parser.add_argument('--d_model', default=256, type=int)
parser.add_argument('--window_size', default=196, type=int,
                    help='the feature map w×h')
parser.add_argument('--nhead', default=8, type=int)
parser.add_argument('--num_decoder_layers', default=2, type=int)
parser.add_argument('--dim_feedforward', default=2048, type=int)
parser.add_argument('--dropout', default=0.1, type=float)
parser.add_argument('--activation', default='relu', type=str)
parser.add_argument('--channel_self_attention', action='store_true',
                    help='change window self-attention to channel self-attention')


def main():
    args = parser.parse_args()
    print(f'args: {args}')

    # get the time for naming log file
    now = datetime.now()
    month, day, hour = now.month, now.day, now.hour

    # fix the seed for reproducibility
    seed = args.rand_seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # prepare dataset
    train_dataset = build_dataset(args, is_train=True)
    sampler_train = torch.utils.data.RandomSampler(train_dataset)
    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)
    train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_sampler=batch_sampler_train)

    # prepare model
    device = torch.device('cuda')
    model = build_semantics_model(args).to(device)

    criterion = build_criterion(args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    print("Start Train")
    for i in range(args.epochs):
        model.train()
        for image, target in train_dataloader:
            image = image.to(device)


            label = label.to(torch.int64).to(device)
            class_feature = class_feature.type(torch.float32).to(device)

            semantics_vector, text_align_vector, class_logits = model(image)
            loss = criterion(text_align_vector, class_logits, class_feature, label, semantics_vector)

            total_loss = loss['total_loss']
            class_loss = loss['class_loss']
            relatedness_loss = loss['relatedness_loss']
            optimizer.zero_grad()
            total_loss.backward()
            # if args.max_norm > 0:  # 详细弄明白之后再用
            #     torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
            optimizer.step()

            total_loss_value = total_loss.item()
            class_loss_value = class_loss.item()
            relatedness_loss_value = relatedness_loss.item()
            print(f"Epoch {i}, Total Loss {total_loss_value} Class Loss {class_loss_value} Relatedness loss {relatedness_loss_value}")

    os.makedirs(args.checkpoint_save_path, exist_ok=True)
    checkpoint_save_path = os.path.join(args.checkpoint_save_path,
                                        f'{month}_{day}_{hour}_{args.desc}_checkpoint.pth')
    torch.save({
        'model': model.state_dict(),
        'args': args,
    }, checkpoint_save_path)


if __name__ == '__main__':
    main()

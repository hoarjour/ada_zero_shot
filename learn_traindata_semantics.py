import argparse
import torch
import numpy as np
import random
import os
import time

from torch.utils.data import DataLoader
from datetime import datetime, timedelta

from datasets import build_dataset
from models import build_semantics_model, build_criterion, build_new_semantics_model
from utils.misc import collate_fn_new, collate_fn, seconds_to_hms, MetricLogger

parser = argparse.ArgumentParser(description="ZSL")

parser.add_argument('--dataset', help='choose between APY, AWA2, AWA1, CUB, SUN', default='CUB', type=str)
parser.add_argument('--epochs', default=300, type=int)
parser.add_argument('--lr', default=0.0001, type=float)
parser.add_argument('--lr_backbone', default=0.00001, type=float)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--rand_seed', default=42, type=int)
parser.add_argument('--lr_drop', default=50, type=int)
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--num_workers', default=2, type=int)

parser.add_argument('--desc', default="default", type=str)

parser.add_argument("--class_feature_type", type=str, default="w2v", choices=['clip', 'w2v'])
parser.add_argument('--markers', default=9, type=int)
parser.add_argument('--compactness', default=0.001, type=float)
parser.add_argument('--bbox_area_threshold', default=0.001, type=float,
                    help='the bbox area ratio threshold compared to whole image area')

parser.add_argument('--eval_every_n_epoch', default=1, type=int)
parser.add_argument('--save_every_n_epoch', default=20, type=int)

# Matcher
parser.add_argument('--set_cost_bbox', default=5, type=float,
                    help="L1 box coefficient in the matching cost")
parser.add_argument('--set_cost_giou', default=2, type=float,
                    help="giou box coefficient in the matching cost")

# Loss coef
parser.add_argument('--class_loss_coef', default=1, type=float)
parser.add_argument('--relatedness_loss_coef', default=0, type=float)
parser.add_argument('--penalty_coef', default=0, type=float)
parser.add_argument('--box_loss_coef', default=1, type=float)
parser.add_argument('--giou_loss_coef', default=0, type=float)

# freeze model
parser.add_argument('--freeze_backnone', action='store_true')

# model args
parser.add_argument('--backbone', default='resnet50', type=str)
parser.add_argument('--return_interm_layers', action='store_true')
parser.add_argument('--dilation', action='store_true')
parser.add_argument('--num_queries', default=100, type=int)
parser.add_argument('--d_model', default=256, type=int)
parser.add_argument('--position_embedding', default='sine', type=str)
parser.add_argument('--hidden_dim', default=256, type=int)
parser.add_argument('--aux_loss', action='store_true')
parser.add_argument('--window_size', default=196, type=int,
                    help='the feature map w×h')
parser.add_argument('--nhead', default=8, type=int)
parser.add_argument('--enc_layers', default=2, type=int)
parser.add_argument('--dec_layers', default=2, type=int)
parser.add_argument('--dim_feedforward', default=2048, type=int)
parser.add_argument('--dropout', default=0.1, type=float)
parser.add_argument('--activation', default='relu', type=str)
parser.add_argument('--pre_norm', action='store_true')
parser.add_argument('--channel_self_attention', action='store_true',
                    help='change window self-attention to channel self-attention')
parser.add_argument('--use_pretrained_features', action='store_true')


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
    used_collate_fn = collate_fn if not args.use_pretrained_features else collate_fn_new

    train_dataset = build_dataset(args, is_train=True)
    sampler_train = torch.utils.data.RandomSampler(train_dataset)
    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)
    train_dataloader = DataLoader(train_dataset, collate_fn=used_collate_fn, batch_sampler=batch_sampler_train,
                                  num_workers=args.num_workers)

    val_dataset = build_dataset(args, is_train=False)
    val_sampler = torch.utils.data.SequentialSampler(val_dataset)
    val_dataloader = DataLoader(val_dataset, args.batch_size, sampler=val_sampler,
                                drop_last=False, collate_fn=used_collate_fn, num_workers=args.num_workers)

    # prepare model
    device = torch.device('cuda')
    model = build_new_semantics_model(args).to(device)

    criterion = build_criterion(args)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    print("Start Train")
    checkpoint_save_dir = f'./data/{args.dataset}/checkpoints'
    os.makedirs(checkpoint_save_dir, exist_ok=True)

    for i in range(args.epochs):
        model.train()
        metric_logger = MetricLogger(delimiter="  ")
        header = 'Epoch: [{}]'.format(i)
        print_freq = 5
        epoch_start = time.time()
        for image, targets in metric_logger.log_every(train_dataloader, print_freq, header):
            if args.use_pretrained_features:
                image = torch.stack(image)
            image = image.to(device)
            for t in targets:
                t["class_feature"] = t["class_feature"].to(device)
                t["boxes"] = t["boxes"].to(device)
                t["original_attr"] = t["original_attr"].to(device)

            semantics_vector, text_align_vector, class_logits, boxes_output = model(image)
            loss = criterion(semantics_vector, text_align_vector, class_logits, boxes_output, targets)

            total_loss = loss['total_loss']
            optimizer.zero_grad()
            total_loss.backward()
            # if args.max_norm > 0:  # 详细弄明白之后再用
            #     torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
            optimizer.step()

            loss_value_dict = {k: v.item() for k, v in loss.items()}
            metric_logger.update(**loss_value_dict)

        train_epoch_cost = time.time() - epoch_start
        train_epoch_cost = seconds_to_hms(train_epoch_cost)
        print("Averaged stats:", metric_logger)
        print(f"epoch {i} Train 消耗时间 f{train_epoch_cost}")

        lr_scheduler.step()

        # val
        print("Start Val")
        val_start = time.time()
        model.eval()
        if (i + 1) % args.eval_every_n_epoch == 0 or i + 1 == args.epochs:
            all_val_images = len(val_dataset)
            correct_num = 0
            metric_logger = MetricLogger(delimiter="  ")
            header = 'Test:'
            print_freq = 5
            for image, targets in metric_logger.log_every(val_dataloader, print_freq, header):
                image = image.to(device)
                for t in targets:
                    t["class_feature"] = t["class_feature"].to(device)
                    t["boxes"] = t["boxes"].to(device)
                    t["original_attr"] = t["original_attr"].to(device)
                with torch.no_grad():
                    semantics_vector, text_align_vector, class_logits, boxes_output = model(image)
                loss = criterion(semantics_vector, text_align_vector, class_logits, boxes_output, targets)
                loss_value_dict = {k: v.item() for k, v in loss.items()}
                metric_logger.update(**loss_value_dict)

                _, predicted = torch.max(class_logits.cpu(), -1)
                gt_labels = torch.Tensor([t['label'] for t in targets]).to(torch.int64)
                correct_num += (predicted == gt_labels).sum().item()

            acc = correct_num / all_val_images
            print(f'Val acc : {acc}')
        val_epoch_cost = time.time() - val_start
        val_epoch_cost = seconds_to_hms(val_epoch_cost)
        print(f"epoch {i} Val 消耗时间 f{val_epoch_cost}")

        if (i + 1) % args.save_every_n_epoch == 0 or i + 1 == args.epochs:
            checkpoint_save_path = os.path.join(checkpoint_save_dir,
                                                f'{month}_{day}_{hour}_{args.desc}_epoch_{i}_checkpoint.pth')
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': i,
                'args': args,
            }, checkpoint_save_path)

    checkpoint_save_path = os.path.join(checkpoint_save_dir,
                                        f'{month}_{day}_{hour}_{args.desc}_final_checkpoint.pth')
    torch.save({
        'model': model.state_dict(),
        'args': args,
    }, checkpoint_save_path)


if __name__ == '__main__':
    main()

import torch
import torch.nn as nn
from models.matcher import build_matcher
import torch.nn.functional as F
from utils import box_ops
EPS=1e-8

class Criterion:

    def __init__(self, class_loss_coef=1, relatedness_loss_coef=1, penalty_coef=1, box_loss_coef=1, giou_loss_coef=1, matcher=None):
        self.class_loss_coef = class_loss_coef
        self.relatedness_loss_coef = relatedness_loss_coef
        self.penalty_coef = penalty_coef
        self.box_loss_coef = box_loss_coef
        self.giou_loss_coef = giou_loss_coef
        self.matcher = matcher

        self.softmax = nn.Softmax(dim=1)

    def cal_class_loss(self, class_logits, label):
        criterion = nn.CrossEntropyLoss()
        class_loss = criterion(class_logits, label)
        return class_loss

    def cal_relatedness_loss(self, text_align_vector, class_feature):
        criterion = nn.MSELoss()
        relatedness_loss = criterion(text_align_vector, class_feature)
        return relatedness_loss

    def cal_penalty(self, semantics_vector):
        x = self.softmax(semantics_vector)
        x = torch.mean(x, 0)
        x_ = torch.clamp(x, min=EPS)
        b = x_ * torch.log(x_)
        return b.sum()

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def cal_box_loss(self, boxes_output, targets):
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(boxes_output, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["boxes"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=boxes_output.device)

        idx = self._get_src_permutation_idx(indices)
        src_boxes = boxes_output[idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        loss_bbox = loss_bbox.sum() / num_boxes
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        loss_giou = loss_giou.sum() / num_boxes
        return loss_bbox, loss_giou

    def __call__(self, semantics_vector, text_align_vector, class_logits, boxes_output, targets):
        class_feature = torch.stack([t['class_feature'] for t in targets])
        original_attr = [t['original_attr'] for t in targets]
        labels = torch.Tensor([t['label'] for t in targets]).to(torch.int64).to(boxes_output.device)

        class_loss = self.cal_class_loss(class_logits, labels)
        relatedness_loss = self.cal_relatedness_loss(text_align_vector, class_feature)
        penalty = self.cal_penalty(semantics_vector)
        box_loss, giou_loss = self.cal_box_loss(boxes_output, targets)

        loss_dict = {}
        loss_dict['unscaled_class_loss'] = class_loss
        loss_dict['class_loss'] = class_loss * self.class_loss_coef
        loss_dict['unscaled_relatedness_loss'] = relatedness_loss
        loss_dict['relatedness_loss'] = relatedness_loss * self.relatedness_loss_coef
        loss_dict['unscaled_penalty'] = penalty
        loss_dict['penalty'] = penalty * self.penalty_coef
        loss_dict['unscaled_box_loss'] = box_loss
        loss_dict['box_loss'] = box_loss * self.box_loss_coef
        loss_dict['unscaled_giou_loss'] = giou_loss
        loss_dict['giou_loss'] = giou_loss * self.giou_loss_coef

        loss_dict['total_loss'] = loss_dict['class_loss'] + loss_dict['relatedness_loss'] + loss_dict['penalty'] + loss_dict['box_loss'] + loss_dict['giou_loss']
        return loss_dict


def build_criterion(args):
    matcher = build_matcher(args)
    criterion = Criterion(args.class_loss_coef,
                          args.relatedness_loss_coef,
                          args.penalty_coef,
                          args.box_loss_coef,
                          args.giou_loss_coef,
                          matcher)

    return criterion

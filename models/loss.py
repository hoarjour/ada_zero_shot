import torch
import torch.nn as nn
EPS=1e-8

class Criterion:

    def __init__(self, class_loss_coef=1, relatedness_loss_coef=1, penalty_coef=1):
        self.class_loss_coef = class_loss_coef
        self.relatedness_loss_coef = relatedness_loss_coef
        self.penalty_coef = penalty_coef

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

    def __call__(self, text_align_vector, class_logits, class_feature, label, semantics_vector):
        class_loss = self.cal_class_loss(class_logits, label)
        relatedness_loss = self.cal_relatedness_loss(text_align_vector, class_feature)
        penalty = self.cal_penalty(semantics_vector)

        loss_dict = {}
        loss_dict['unscaled_class_loss'] = class_loss
        loss_dict['class_loss'] = class_loss * self.class_loss_coef
        loss_dict['unscaled_relatedness_loss'] = relatedness_loss
        loss_dict['relatedness_loss'] = relatedness_loss * self.relatedness_loss_coef
        loss_dict['unscaled_penalty'] = penalty
        loss_dict['penalty'] = penalty * self.penalty_coef
        loss_dict['total_loss'] = loss_dict['class_loss'] + loss_dict['relatedness_loss'] + loss_dict['penalty']
        return loss_dict


def build_criterion(args):
    criterion = Criterion(args.class_loss_coef,
                          args.relatedness_loss_coef,
                          args.penalty_coef)

    return criterion

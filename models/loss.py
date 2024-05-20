import torch.nn as nn


class Criterion:

    def __init__(self):
        pass

    def cal_class_loss(self, class_logits, label):
        criterion = nn.CrossEntropyLoss()
        class_loss = criterion(class_logits, label)
        return class_loss

    def cal_relatedness_loss(self, text_align_vector, class_feature):
        criterion = nn.MSELoss()
        relatedness_loss = criterion(text_align_vector, class_feature)
        return relatedness_loss

    def __call__(self, text_align_vector, class_logits, class_feature, label):
        return self.cal_class_loss(class_logits, label), self.cal_relatedness_loss(text_align_vector, class_feature)


def build_criterion(args):
    criterion = Criterion()

    return criterion

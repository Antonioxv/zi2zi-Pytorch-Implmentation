import torch
import torch.nn as nn


class CategoryLoss(nn.Module):
    def __init__(self, category_num):
        super(CategoryLoss, self).__init__()
        emb = nn.Embedding(category_num, category_num)
        emb.weight.data = torch.eye(category_num)
        self.emb = emb
        self.loss = nn.BCEWithLogitsLoss()  # bce

    def forward(self, category_logits, labels):
        target = self.emb(labels)
        return self.loss(category_logits, target)


class BinaryLoss(nn.Module):
    def __init__(self):
        super(BinaryLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, real=True):
        if real:
            labels = torch.ones(logits.shape[0], 1)
        else:
            labels = torch.zeros(logits.shape[0], 1)
        if logits.is_cuda:  # move to gpu when gpu is available
            labels = labels.cuda()
        return self.bce(logits, labels)

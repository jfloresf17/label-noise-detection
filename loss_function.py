import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, y_true, y_pred, treshold=0.5):
        y_pred = (y_pred > treshold).float()
        
        ## Flatten the tensors
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)

        intersection = (y_pred * y_true).sum()
        union = y_pred.sum() + y_true.sum()
        dice = 1 -((2 * intersection + self.smooth) / (union + self.smooth))
        return dice


    def to(self, device):
        super(DiceLoss, self).to(device)
        return self


"""
Lovasz hinge loss in PyTorch
Maxim Berman 2018 ESAT-PSI KU Leuven 
"""

import torch
from torch.autograd import Variable
import torch.nn.functional as F

def LovaszLoss(logits, labels):
    """
    Binary Lovasz hinge loss
        logits: [B, C, H, W] Variable, logits at each pixel (between -\infty and +\infty)
        labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
    """

    if len(labels) == 0:
        return logits.sum() * 0.
    
    ## Flatten the logits and labels
    logits = logits.view(-1)
    labels = labels.view(-1)

    ## Compute the loss
    signs = 2. * labels.float() - 1.
    errors = (1. - logits * Variable(signs))
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    
    p = len(gt_sorted)

    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union

    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]

    loss = torch.dot(F.relu(errors_sorted), Variable(jaccard))

    return loss

def LovaszLossBatch(logits, labels):
    """
    Lovasz Loss aplicado a un batch de imágenes.
        logits: [B, C, H, W] Variable, logits at cada pixel
        labels: [B, H, W] Tensor, etiquetas binarias de ground truth (0 o 1)
    """
    batch_loss = 0.
    for i in range(logits.size(0)):  # Iterar sobre cada imagen en el batch
        loss = LovaszLoss(logits[i], labels[i])
        batch_loss += loss
    
    # Promedio de pérdidas sobre el batch
    batch_loss = batch_loss / logits.size(0) # batch_loss /= logits.size(0)
    return batch_loss
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

def normalize(logit):
    mean = logit.mean(dim=-1, keepdims=True)
    stdv = logit.std(dim=-1, keepdims=True)
    return (logit - mean) / stdv

class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self):
        super(DistillKL, self).__init__()

    def forward(self, y_s, y_t, temp):
        T = temp.cuda()
        
        KD_loss = 0
        KD_loss += nn.KLDivLoss(reduction='batchmean')(F.log_softmax(normalize(y_s)/T, dim=1),
                                F.softmax(normalize(y_t)/T, dim=1)) * T * T
        
        return KD_loss

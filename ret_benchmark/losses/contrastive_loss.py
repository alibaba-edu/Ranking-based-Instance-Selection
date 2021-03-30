from __future__ import absolute_import

import torch,pickle
from torch import nn
from torch.autograd import Variable
import numpy as np
from ret_benchmark.losses.registry import LOSS
from ret_benchmark.utils.log_info import log_info
import os

@LOSS.register("contrastive_loss")
class ContrastiveLoss(nn.Module):
    def __init__(self, cfg, checked_outlier=None):
        super(ContrastiveLoss, self).__init__()
        self.margin = 0.5
        self.checked_outlier=None
        self.iteration=0
        self.name=cfg.NAME

    def forward(self, inputs_col, targets_col, inputs_row, target_row,is_noise=None):

        n = inputs_col.size(0)

        is_batch=(inputs_col.shape[0] == inputs_row.shape[0])
        # Compute similarity matrix
        sim_mat = torch.matmul(inputs_col, inputs_row.t())
        epsilon = 1e-5
        loss = list()

        pos_mask = targets_col.expand(target_row.shape[0], n).t() == target_row.expand(n, target_row.shape[0])
        neg_mask = (~pos_mask) & (sim_mat>self.margin)
        pos_mask = pos_mask & (sim_mat<(1-epsilon))
 
        pos_pair=sim_mat[pos_mask]
        neg_pair=sim_mat[neg_mask]
        
        pos_loss = torch.sum(-pos_pair + 1)
        if len(neg_pair) > 0:
            neg_loss = torch.sum(neg_pair)
        else:
            neg_loss = 0

        if is_batch:
            prefix = "batch_"
        else:
            prefix = "memory_"

        loss = (pos_loss+neg_loss) / n  # / all_targets.shape[1]
        if not is_batch:
            prefix = "xbm_"
            log_info[f"{prefix}loss"] = loss.item()
            self.iteration+=1

        return loss
            

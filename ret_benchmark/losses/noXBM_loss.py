from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
from ret_benchmark.losses.registry import LOSS
from ret_benchmark.utils.log_info import log_info
import numpy as np
import random



@LOSS.register("noXBM_loss")
class noXBMLoss(nn.Module):
    def __init__(self, num_classes,cfg):
        super(noXBMLoss, self).__init__()
        embedding_size=cfg.MODEL.HEAD.DIM
        from pytorch_metric_learning import losses
        loss_name=cfg.LOSSES.NOXBM_LOSS_NAME
        if loss_name=='circle_loss':
            loss_func = losses.CircleLoss(m=0.4, gamma=80)
        elif loss_name=='softtriple_loss':
            K=10 if num_classes<500 else 2
            loss_func = losses.SoftTripleLoss(num_classes, 
                    embedding_size, 
                    centers_per_class=K, 
                    la=20, 
                    gamma=0.1, 
                    margin=0.01)
        elif loss_name=='fastAP_loss':
            loss_func=losses.FastAPLoss(num_bins=10)
        elif loss_name=='nSoftmax_loss':
            loss_func=losses.NormalizedSoftmaxLoss(num_classes, embedding_size, temperature=0.05)
        elif loss_name=='proxyNCA_loss':
            loss_func=losses.ProxyNCALoss(num_classes, embedding_size, softmax_scale=3)
        else:
            print('noXBM loss unknown. Given',loss_name)
            assert False
        self.loss_func=loss_func

    def forward(self, inputs_col, targets_col, inputs_row, target_row,is_noise=None):
        loss=self.loss_func(inputs_col,targets_col)
        return loss

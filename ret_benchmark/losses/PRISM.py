from __future__ import absolute_import

import torch
from torch import nn
from ret_benchmark.losses.registry import LOSS
from ret_benchmark.utils.log_info import log_info


@LOSS.register("PRISM")
class PRISM(nn.Module):
    def __init__(self, cfg):
        super(PRISM, self).__init__()
        self.noise_rate=cfg.NOISE.NOISE_RATE
        self.margin_window=[]
        self.window_size=int(cfg.NOISE.WINDOW_SIZE)
        num_classes=cfg.num_classes
        emd_size=cfg.MODEL.HEAD.DIM
        self.center=torch.zeros(size=[num_classes,emd_size]).cuda()
        self.filled_center=set()
        self.last_target_col=None

    def update_center(self,inputs_row,target_row):
        # only need to update class that is newly inserted
        if self.last_target_col is not None:
            for i in torch.unique(self.last_target_col):
                i=i.item()
                row_mask= (target_row==i)
                self.center[i]=torch.mean(inputs_row[row_mask],dim=0)
                self.filled_center.add(i)

    def forward(self, inputs_col, targets_col, inputs_row, target_row,is_noise=None):
        n = inputs_col.size(0)
        if inputs_row.shape[0]==0:
            return 0,torch.ones_like(targets_col,dtype=bool).cuda()
        is_batch=(targets_col is target_row)


        epsilon = 1e-6
       
        if not is_batch:
            with torch.no_grad():
                self.update_center(inputs_row,target_row)
                C=[]
                for i in range(n):
                    selected_cls=targets_col[i].item()
                    if selected_cls not in self.filled_center:
                        C.append(1)
                    else:
                        all_sim_exp=torch.exp(torch.mm(self.center,inputs_col[i].view(-1,1))).view(-1)
                        softmax_loss=all_sim_exp[selected_cls]/(torch.sum(all_sim_exp)+epsilon)
                        C.append(softmax_loss)
                C=torch.Tensor(C)
                idx_sorted = torch.argsort(C)
                to_remove=idx_sorted[:int(self.noise_rate*len(C))]


                #update window
                if not torch.isnan(C[to_remove[-1]]) and C[to_remove[-1]]!=1.0:
                    self.margin_window.append(C[to_remove[-1]].item())
                    self.margin_window=self.margin_window[-self.window_size:]

                if len(self.margin_window)>0:
                    keep_bool=(C>sum(self.margin_window)/len(self.margin_window)).cpu()
                    if torch.any(keep_bool)==False:
                        keep_bool=torch.ones_like(targets_col,dtype=bool)
                        keep_bool[to_remove]=False
                        self.margin_window=self.margin_window[-1:]
                        log_info[f"PRISM_threshold"] = C[to_remove[-1]]
                    else:
                        log_info[f"PRISM_threshold"] = sum(self.margin_window)/len(self.margin_window)
                else:
                    keep_bool=torch.ones_like(targets_col,dtype=bool)
                    keep_bool[to_remove]=False
                    log_info[f"PRISM_threshold"] = C[to_remove[-1]]

        if is_noise is not None:
            is_noise=is_noise[keep_bool]
            log_info[f"PRISM_pure_rate"] = 1-torch.mean(is_noise.float()).item()
        self.last_target_col=targets_col[keep_bool].clone()
        log_info[f"PRISM_remove_ratio"] = torch.mean(keep_bool.float()).item()
        return 0, keep_bool



from .contrastive_loss import ContrastiveLoss
from .registry import LOSS
from .noXBM_loss import noXBMLoss
from .memory_contrastive_loss_w_PRISM import MemoryContrastiveLossPRISM
from .PRISM import PRISM
def build_loss(loss_name,num_class,cfg):
    assert loss_name in LOSS, f"loss name {loss_name} is not registered in registry"
    if 'noXBM' in loss_name :
        loss=LOSS[loss_name](num_class,cfg)
        if len(list(loss.parameters()))>0: 
            import torch
            loss_optimizer = torch.optim.Adam(loss.parameters(), lr=cfg.LOSSES.CENTER_LR)
            print('loss has parameters {}'.format([i[0] for i in loss.named_parameters()]))
            return loss,loss_optimizer
        return loss
    return LOSS[loss_name](cfg)

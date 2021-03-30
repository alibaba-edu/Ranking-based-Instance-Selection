import torch

def build_optimizer(cfg, model):
    optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(
        list(model.parameters()),
        lr=cfg.SOLVER.BASE_LR,
        weight_decay=cfg.SOLVER.WEIGHT_DECAY,
    )
    return optimizer


def build_lr_scheduler(cfg, optimizer):
    if cfg.SOLVER.LR_SCHEDULAR=='cos':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(optimizer,T_max=cfg.SOLVER.MAX_ITERS)
    else:
        print('LR schedular not found')
        exit()
    return scheduler